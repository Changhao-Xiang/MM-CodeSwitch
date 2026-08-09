from typing import Any, List, Optional, Tuple, Union

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.siglip import SiglipVisionConfig, SiglipVisionModel

from model.codeswitch import extract_patch_features_by_indices
from model.llava_next_qwen2.configuration_llava_next_qwen2 import LlavaNextQwen2Config
from model.projectors import PatchMerger


class LlavaNextQwen2ForCausalLM(PreTrainedModel):
    config_class = LlavaNextQwen2Config
    main_input_name = "inputs_embeds"
    base_model_prefix = "language_model"
    _no_split_modules = ["CLIPVisionModel", "SiglipVisionModel", "Qwen2DecoderLayer"]
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: LlavaNextQwen2Config,
        vision_model: Optional[PreTrainedModel] = None,
        language_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)
        self.config = config

        # Init Vision Encoder
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if "clip" in config.vision_config.model_type:
                assert isinstance(config.vision_config, CLIPVisionConfig)
                self.vision_model = CLIPVisionModel(config.vision_config)
            elif "siglip" in config.vision_config.model_type:
                assert isinstance(config.vision_config, SiglipVisionConfig)
                self.vision_model = SiglipVisionModel(config.vision_config)
            else:
                raise ValueError(f"Unsupported vision tower: {config.vision_config.model_type}")

        # Init Multi-Modal Projector
        self.projector = PatchMerger(config)

        # Init Language Model
        if language_model is not None:
            self.language_model = language_model
        else:
            assert isinstance(config.llm_config, PretrainedConfig)
            self.language_model = Qwen2ForCausalLM(config.llm_config)

        self.pad_token_id = 151643  # <|endoftext|> of Qwen2Tokenizer
        self.image_token_id = config.image_token_id

        # Config multimodal code-switching
        if not hasattr(self.config, "switch_embeds"):
            self.config.switch_embeds = False

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def tie_weights(self):
        if hasattr(self.language_model, "tie_weights"):
            self.language_model.tie_weights()

    def _merge_inputs_embeds_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        image_features = image_features.to(dtype=dtype, device=device)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_id) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_id
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        return final_embedding

    def _merge_inputs_embeds_with_mmcs_obj_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        num_image_tiles_batch: List[int],
        patch_indices_list_batch: List[Any | None],
    ):
        # Calculate the starting index for each sample's image features
        image_feature_start_idx = 0

        for i, patch_indices_list in enumerate(patch_indices_list_batch):
            num_tiles = num_image_tiles_batch[i]

            # Get image features for this sample (all tiles)
            if num_tiles > 0:
                sample_image_features = image_features[image_feature_start_idx : image_feature_start_idx + num_tiles]
                # Flatten tiles: (num_tiles, num_patches_per_tile, hidden_size) -> (total_patches, hidden_size)
                sample_image_features = sample_image_features.view(-1, sample_image_features.shape[-1])
            else:
                sample_image_features = None

            # Current sample contains objects to switch embeddings
            if patch_indices_list is not None and len(patch_indices_list) > 0 and sample_image_features is not None:
                # Extract features for each object separately
                obj_features_list = []
                for patch_indices in patch_indices_list:
                    obj_features = extract_patch_features_by_indices(sample_image_features, patch_indices)
                    obj_features_list.append(obj_features)

                # Concatenate all object features
                obj_features = torch.cat(obj_features_list, dim=0)

                if self.config.add_whole_image:
                    obj_features = torch.cat([sample_image_features, obj_features], dim=0)

                inputs_embeds[i] = self._merge_inputs_embeds_with_image_features(
                    obj_features.unsqueeze(0), inputs_embeds[i].unsqueeze(0), input_ids[i].unsqueeze(0)
                )
            # Current sample contains caption only
            elif sample_image_features is not None:
                inputs_embeds[i] = self._merge_inputs_embeds_with_image_features(
                    sample_image_features.unsqueeze(0),
                    inputs_embeds[i].unsqueeze(0),
                    input_ids[i].unsqueeze(0),
                )

            # Update the starting index for the next sample
            image_feature_start_idx += num_tiles

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_image_tiles_batch: Optional[List[int]] = None,
        patch_indices_list_batch: Optional[List[Any | None]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Extract the input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Merge text and images
        if pixel_values is not None:
            assert (input_ids == self.config.image_token_id).any()
            # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Vis_Embed_Dim]
            image_features = self.vision_model(pixel_values.to(inputs_embeds.dtype)).last_hidden_state
            # Skip the [CLS] token for clip, note that siglip has no [CLS] token
            if "clip" in self.config.vision_config.model_type:
                image_features = image_features[:, 1:, :]

            # [Batch_Size, Num_Patches, Vis_Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
            image_features = self.projector(image_features)

            # Merge the embeddings of the text tokens and the image tokens
            if (
                self.config.switch_embeds
                and patch_indices_list_batch is not None
                and num_image_tiles_batch is not None
                and any(patch_indices_list is not None for patch_indices_list in patch_indices_list_batch)
            ):
                inputs_embeds = self._merge_inputs_embeds_with_mmcs_obj_features(
                    image_features, inputs_embeds, input_ids, num_image_tiles_batch, patch_indices_list_batch
                )
            else:  # No switch embeds
                inputs_embeds = self._merge_inputs_embeds_with_image_features(image_features, inputs_embeds, input_ids)

        outputs = self.language_model(attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

        return outputs

    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Vis_Embed_Dim]
            image_features = self.vision_model(pixel_values.to(inputs_embeds.dtype)).last_hidden_state
            # Skip the [CLS] token for clip, note that siglip has no [CLS] token
            if "clip" in self.config.vision_config.model_type:
                image_features = image_features[:, 1:, :]
            # [Batch_Size, Num_Patches, Vis_Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
            image_features = self.projector(image_features)
            # Merge the embeddings of the text tokens and the image tokens
            inputs_embeds = self._merge_inputs_embeds_with_image_features(image_features, inputs_embeds, input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        return outputs
