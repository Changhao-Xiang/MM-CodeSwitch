import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import json
import os
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from common.kosmos_tokens import post_process_kosmos_generation
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from model import *
from model.dynamic_resolution import dynamic_preprocess

DEFAULT_IMAGE_TOKEN = "<image>"

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

# if version.parse(torch.__version__) >= version.parse("2.1.2"):
#     best_fit_attn_implementation = "sdpa"
# else:
#     best_fit_attn_implementation = "eager"
best_fit_attn_implementation = "flash_attention_2"
# best_fit_attn_implementation = "eager"


def infer_language_model_type(pretrained: str) -> str:
    pretrained_lower = pretrained.lower()
    if "qwen2" in pretrained_lower:
        return "qwen2"
    if "qwen3" in pretrained_lower:
        return "qwen3"
    if "llama" in pretrained_lower:
        return "llama"
    if "vicuna" in pretrained_lower:
        return "vicuna"

    config_path = os.path.join(pretrained, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        llm_config = config.get("llm_config", {})
        architectures = [arch.lower() for arch in llm_config.get("architectures", [])]
        model_type = str(llm_config.get("model_type", "")).lower()
        if any("qwen2" in arch for arch in architectures) or model_type == "qwen2":
            return "qwen2"
        if any("qwen3" in arch for arch in architectures) or model_type == "qwen3":
            return "qwen3"
        if any("llama" in arch for arch in architectures) or "llama" in model_type:
            return "llama"
        if any("vicuna" in arch for arch in architectures) or "vicuna" in model_type:
            return "vicuna"

    raise ValueError(f"Unsupported language model: {pretrained}")


@register_model("llava_mmcs")
class Llava_mmcs(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "checkpoints/siglip-qwen2.5-3b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        use_cache=True,
        max_length=4096,
        max_pixels=728 * 28 * 28,
        kosmos_postprocess: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        self.kosmos_postprocess = kosmos_postprocess

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        language_model_type = infer_language_model_type(pretrained)

        if "navit" in pretrained:
            self.dynamic_resolution = "native"
            if language_model_type == "qwen2":
                config_class = LlavaQwen2NativeResConfig
                model_class = LlavaQwen2NativeResForCausalLM
                processor_class = LlavaQwen2NativeResProcessor
        else:
            self.dynamic_resolution = "tile"
            if language_model_type == "qwen2":
                config_class = LlavaNextQwen2Config
                model_class = LlavaNextQwen2ForCausalLM
                processor_class = LlavaNextQwen2Processor
            elif language_model_type == "qwen3":
                config_class = LlavaNextQwen3Config
                model_class = LlavaNextQwen3ForCausalLM
                processor_class = LlavaNextQwen3Processor
            elif language_model_type == "llama":
                config_class = LlavaNextLlamaConfig
                model_class = LlavaNextLlamaForCausalLM
                processor_class = LlavaNextLlamaProcessor
            elif language_model_type == "vicuna":
                config_class = LlavaNextVicunaConfig
                model_class = LlavaNextVicunaForCausalLM
                processor_class = LlavaNextVicunaProcessor

        self._config = config_class.from_pretrained(pretrained)
        self._processor = processor_class.from_pretrained(pretrained, use_fast=False)
        if "navit" in pretrained:
            self._processor.image_processor.max_pixels = max_pixels
        self._processor.tokenizer.padding_side = "left"
        # self._processor.tokenizer.eos_token = "<|im_end|>"

        self._model = model_class.from_pretrained(
            pretrained,
            config=self._config,
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation,
            # device_map=device_map,
        )

        self._max_length = max_length
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info(
                    "Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0"
                )

            if (
                accelerator.distributed_type == DistributedType.FSDP
                or accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._processor.tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        if hasattr(self.tokenizer, "eot_token_id"):
            # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
            return self.tokenizer.eot_token_id
        else:
            return self.tokenizer.pad_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for LLaVA")

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            # Task configs are shared across requests.  Work on a copy because
            # popping ``until`` from the original silently changes later batches.
            gen_kwargs = copy.deepcopy(all_gen_kwargs[0])

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                task_until = gen_kwargs.pop("until")
                if isinstance(task_until, str):
                    task_until = [task_until]
                elif not isinstance(task_until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(task_until)}"
                    )
                until.extend(task_until)

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

            # encode, pad, and truncate contexts for this batch
            if flattened_visuals:
                if self.dynamic_resolution == "tile":
                    processed_visuals = []
                    num_image_tiles = []
                    for visual in flattened_visuals:
                        processed_visual, target_aspect_ratio = dynamic_preprocess(
                            visual,
                            tile_size=self._config.vision_config.image_size,
                            max_num_tiles=self._config.vision_config.max_num_tiles,
                            use_thumbnail=True,
                        )
                        processed_visuals.extend(processed_visual)
                        num_image_tiles.append(len(processed_visual))
                    image_tensor = self._processor.image_processor(processed_visuals, return_tensors="pt")[
                        "pixel_values"
                    ]
                elif self.dynamic_resolution == "native":
                    for i in range(len(flattened_visuals)):
                        while flattened_visuals[i].width < 28 or flattened_visuals[i].height < 28:
                            flattened_visuals[i] = flattened_visuals[i].resize(
                                (flattened_visuals[i].width * 2, flattened_visuals[i].height * 2)
                            )
                    visual_processed = self._processor.image_processor.preprocess(
                        flattened_visuals, return_tensors="pt"
                    )
                    image_tensor = visual_processed["pixel_values"]
                    image_grid_thw = visual_processed["image_grid_thw"]
                    num_image_tiles = [int(thw[1] * thw[2] // 4) for thw in image_grid_thw]
                else:
                    num_image_tiles = [1] * len(flattened_visuals)
                    image_tensor = self._processor.image_processor(flattened_visuals, return_tensors="pt")[
                        "pixel_values"
                    ]
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            messages_list = []
            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = (
                        [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    )
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                messages_list.append(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ]
                )

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            if self._processor.tokenizer.chat_template is None:
                self._processor.tokenizer.chat_template = "{%- for message in messages %}{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}\n"

            input_text_list = [
                self._processor.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, return_tensors="pt"
                )
                for messages in messages_list
            ]
            if image_tensor is not None and len(image_tensor) != 0:
                input_text_list = [
                    self._processor.expand_image_tokens(text, num_image_tiles=num_image_tiles)
                    for text in input_text_list
                ]
            input_ids_list = [
                self._processor.tokenizer(text, return_tensors="pt").input_ids.view(-1) for text in input_text_list
            ]

            pad_token_ids = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            try:
                if self.dynamic_resolution == "native":
                    output_ids = self.model.generate(
                        input_ids,
                        pixel_values=image_tensor,
                        image_grid_thw=image_grid_thw,
                        attention_mask=attention_masks,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    output_ids = self.model.generate(
                        input_ids,
                        pixel_values=image_tensor,
                        attention_mask=attention_masks,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                # KOSMOS coordinate/markup tokens are regular added tokens, not
                # tokenizer special tokens.  Keeping tokenizer special tokens
                # here leaks Qwen chat delimiters (for example ``A.<|im_end|>``)
                # into benchmark answers without preserving anything needed for
                # grounding.
                text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                if self.kosmos_postprocess is not None:
                    text_outputs = [
                        post_process_kosmos_generation(
                            text,
                            mode=self.kosmos_postprocess,
                            num_bins=getattr(self._config, "coordinate_token_bins", 32),
                        )
                        for text in text_outputs
                    ]
                # Some generation backends do not consume task-level `until`
                # strings as stopping criteria.  Match lm-eval semantics by
                # trimming the decoded response at the first requested stop.
                text_outputs = [
                    text[: min([text.find(term) for term in until if term and term in text] or [len(text)])].strip()
                    for text in text_outputs
                ]
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")
