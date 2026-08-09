import copy
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import AddedToken

from common.kosmos_tokens import KOSMOS_DEFAULT_NUM_BINS, kosmos_tokens
from model.constants import IGNORE_INDEX, IMAGE_TOKEN, ROLE_MAP
from model.dynamic_resolution import dynamic_preprocess


class LlavaNextQwen2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "Qwen2Tokenizer"

    def __init__(self, image_processor=None, tokenizer=None):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        self.num_new_tokens = 0

        if not hasattr(tokenizer, "image_token_id"):
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            self.num_new_tokens += tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        else:
            self.image_token_id = tokenizer.image_token_id

        self.coordinate_token_bins = KOSMOS_DEFAULT_NUM_BINS
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        super().__init__(image_processor, tokenizer)

    def enable_coordinate_tokens(self, num_bins: int = KOSMOS_DEFAULT_NUM_BINS) -> int:
        coordinate_tokens = [
            AddedToken(token, normalized=False, special=False)
            for token in kosmos_tokens(num_bins)
            if self.tokenizer.convert_tokens_to_ids(token) == self.tokenizer.unk_token_id
        ]
        num_added = self.tokenizer.add_tokens(coordinate_tokens)
        self.num_new_tokens += num_added
        self.coordinate_token_bins = num_bins
        return num_added

    def __call__(
        self,
        text: str | List[str],
        images: Optional[Image.Image | List[Image.Image]] = None,
        padding: str = "longest",
        truncation: bool = True,
    ):
        if images is None:  # Pure text input
            inputs = self.tokenizer(text, return_tensors="pt", padding=padding, truncation=truncation)
            return inputs
        else:  # Image-text input
            if isinstance(images, Image.Image):
                images = [images]
            dynamic_processed_images = []
            num_image_tiles = []
            for image in images:
                processed_images, target_aspect_ratio = dynamic_preprocess(
                    image, tile_size=self.image_processor.size["height"], max_num_tiles=4
                )
                dynamic_processed_images.extend(processed_images)
                num_image_tiles.append(len(processed_images))
            pixel_values = self.image_processor(dynamic_processed_images, return_tensors="pt").pixel_values

            if isinstance(text, str):
                text = [text]
            processed_text = []
            for t in text:
                # Prepend a `self.image_seq_length` number of image tokens to the prompt
                if IMAGE_TOKEN not in t:
                    input_strings = self.add_image_tokens(t, num_image_tiles)
                elif self.image_seq_length * IMAGE_TOKEN not in t:
                    input_strings = self.expand_image_tokens(t, num_image_tiles)
                else:
                    input_strings = t
                processed_text.append(input_strings)

            # Returns the input_ids and attention_mask as PyTorch tensors
            inputs = self.tokenizer(processed_text, return_tensors="pt", padding=padding, truncation=truncation)

            return_data = {"pixel_values": pixel_values, **inputs}

            return return_data

    def preprocess(
        self,
        conversations: List[Dict],
        num_image_tiles: int = 1,
        default_system_prompt: str = "You are a helpful assistant.",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Modify chat template so that it won't include system message every time apply
        tokenizer = copy.deepcopy(self.tokenizer)
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        input_ids = []
        label_ids = []

        if conversations[0]["from"] != "system":
            conversations.insert(0, {"from": ROLE_MAP["system"], "value": default_system_prompt})

        for conversation in conversations:
            role, content = conversation["from"], conversation["value"]
            if role not in ROLE_MAP.values():
                role = ROLE_MAP[role]
            if IMAGE_TOKEN in content:
                content = self.expand_image_tokens(content, [num_image_tiles])
            ids = tokenizer.apply_chat_template([{"role": role, "content": content}])
            input_ids.extend(ids)

            if role in [ROLE_MAP["human"], ROLE_MAP["system"]]:
                label_ids.extend([IGNORE_INDEX] * len(ids))
            else:
                label_ids.extend(ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return input_ids, label_ids

    def add_image_tokens(self, content: str, num_image_tiles: List[int]) -> str:
        assert len(num_image_tiles) == 1
        return f"{IMAGE_TOKEN * self.image_seq_length * num_image_tiles[0]}\n{content}"

    def expand_image_tokens(self, content: str, num_image_tiles: List[int]) -> str:
        assert IMAGE_TOKEN in content
        for n in num_image_tiles:
            content = content.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length * n, 1)
        return content
