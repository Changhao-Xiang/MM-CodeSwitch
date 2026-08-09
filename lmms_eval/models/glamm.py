import re
import sys
from datetime import timedelta
from typing import List, Optional, Tuple, Union

sys.modules.setdefault("deepspeed", None)

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models._grounding_model_utils import (
    DEFAULT_LOCAL_CLIP_VISION_TOWER,
    allow_transformers_duplicate_registration,
    disable_broken_deepspeed_import,
    ensure_repo_first,
    get_doc_region_box_xyxy,
    get_task_max_new_tokens,
    inject_region_token,
    install_lightweight_mmcv_compat,
    install_repo_namespace_packages,
    is_bbox_prediction_task,
    normalize_text_only_output,
    purge_conflicting_top_level_modules,
    route_clip_vision_tower,
)

DEFAULT_GLAMM_REPO_PATH = "third_party/GLaMM"
DEFAULT_GLAMM_CKPT_PATH = "checkpoints/GLaMM-FullScope"
_COORD_RE = re.compile(
    r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
)


def _mask_to_normalized_bbox(pred_masks):
    """Convert GLaMM's first predicted segmentation mask to a normalized xyxy box."""
    if not pred_masks or pred_masks[0] is None or pred_masks[0].numel() == 0:
        return None

    masks = pred_masks[0]
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    mask = masks[0] > 0
    coordinates = mask.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return None

    height, width = mask.shape[-2:]
    y1, x1 = coordinates.min(dim=0).values
    y2, x2 = coordinates.max(dim=0).values + 1
    return "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
        x1.item() / width,
        y1.item() / height,
        x2.item() / width,
        y2.item() / height,
    )


@register_model("glamm")
class Glamm(lmms):
    def __init__(
        self,
        pretrained: str = DEFAULT_GLAMM_CKPT_PATH,
        glamm_repo_path: str = DEFAULT_GLAMM_REPO_PATH,
        vision_tower_path: Optional[str] = DEFAULT_LOCAL_CLIP_VISION_TOWER,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        conv_type: str = "llava_v1",
        image_size: int = 1024,
        model_max_length: int = 512,
        torch_dtype: str = "bfloat16",
        use_mm_start_end: bool = True,
        add_region_feature: bool = True,
        generate_masks: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        if not torch.cuda.is_available():
            raise RuntimeError("GLaMM inference requires CUDA because the official model code calls .cuda().")

        disable_broken_deepspeed_import()
        route_clip_vision_tower(vision_tower_path)
        glamm_repo_path = ensure_repo_first(glamm_repo_path)
        purge_conflicting_top_level_modules(glamm_repo_path, ("model", "tools", "eval", "mmdet"))
        ensure_repo_first(glamm_repo_path)
        install_repo_namespace_packages(glamm_repo_path, ("model", "tools", "eval"))
        install_lightweight_mmcv_compat()
        allow_transformers_duplicate_registration()

        from eval.utils import grounding_image_ecoder_preprocess
        from tools.utils import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )

        from model.GLaMM import GLaMMForCausalLM
        from model.llava import conversation as conversation_lib
        from model.llava.mm_utils import tokenizer_image_token
        from model.SAM.utils.transforms import ResizeLongestSide

        self.GLaMMForCausalLM = GLaMMForCausalLM
        self.ResizeLongestSide = ResizeLongestSide
        self.conversation_lib = conversation_lib
        self.tokenizer_image_token = tokenizer_image_token
        self.grounding_image_ecoder_preprocess = grounding_image_ecoder_preprocess
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        torch.cuda.set_device(self._device)
        self.dtype = getattr(torch, torch_dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
            local_files_only=True,
        )
        self._tokenizer.pad_token = self._tokenizer.unk_token
        seg_token_idx = self._tokenizer.convert_tokens_to_ids("[SEG]")
        bbox_token_idx = self._tokenizer.convert_tokens_to_ids("<bbox>")

        self._model = GLaMMForCausalLM.from_pretrained(
            pretrained,
            low_cpu_mem_usage=True,
            seg_token_idx=seg_token_idx,
            bbox_token_idx=bbox_token_idx,
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        self._model.config.eos_token_id = self._tokenizer.eos_token_id
        self._model.config.bos_token_id = self._tokenizer.bos_token_id
        self._model.config.pad_token_id = self._tokenizer.pad_token_id

        self._model.get_model().initialize_vision_modules(self._model.get_model().config)
        self._model = self._model.to(device=self._device, dtype=self.dtype)
        self._model.get_model().get_vision_tower().to(device=self._device, dtype=self.dtype)
        self._model.eval()

        self._config = self._model.config
        self._image_processor = CLIPImageProcessor.from_pretrained(self._config.vision_tower, local_files_only=False)
        self.transform = ResizeLongestSide(image_size)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        self.batch_size_per_gpu = int(batch_size)
        self.conv_type = conv_type
        self.image_size = image_size
        self.model_max_length = model_max_length
        self.use_mm_start_end = use_mm_start_end
        self.add_region_feature = add_region_feature
        self.generate_masks = generate_masks

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model_max_length

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
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        return encoding[-left_truncate_len:] if left_truncate_len else encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("GLaMM lmms_eval adapter currently supports generate_until only.")

    def _build_prompt(self, context):
        conv = self.conversation_lib.conv_templates[self.conv_type].copy()
        conv.messages = []
        if self.DEFAULT_IMAGE_TOKEN not in context:
            context = f"The {self.DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n{context}"
        if self.use_mm_start_end:
            replace_token = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
            context = context.replace(self.DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], context)
        conv.append_message(conv.roles[1], "")
        return conv.get_prompt()

    def _prepare_images(self, image: Image.Image):
        image_np = np.asarray(image.convert("RGB"))
        original_size_list = [image_np.shape[:2]]

        global_image = self._prepare_global_image(image_np)

        grounding_image = self.transform.apply_image(image_np)
        resize_list = [grounding_image.shape[:2]]
        grounding_image = torch.from_numpy(grounding_image).permute(2, 0, 1).contiguous()
        grounding_image = self.grounding_image_ecoder_preprocess(grounding_image)
        grounding_image = grounding_image.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        return global_image, grounding_image, resize_list, original_size_list

    def _prepare_global_image(self, image):
        image_np = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else image
        global_image = self._image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        return global_image.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def _raw_box_to_normalized(self, raw_box, raw_w, raw_h):
        x1 = max(0.0, min(float(raw_w), raw_box[0]))
        x2 = max(0.0, min(float(raw_w), raw_box[2]))
        y1 = max(0.0, min(float(raw_h), raw_box[1]))
        y2 = max(0.0, min(float(raw_h), raw_box[3]))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return [x1 / raw_w, y1 / raw_h, x2 / raw_w, y2 / raw_h]

    def _prepare_region_prompt_and_bboxes(self, context, image: Image.Image, fallback_box=None):
        if not self.add_region_feature:
            return context, None

        raw_w, raw_h = image.size
        bboxes = []

        def replace(match):
            coords = [float(match.group(i)) for i in range(1, 5)]
            if all(0 <= value <= 1 for value in coords):
                raw_box = [coords[0] * raw_w, coords[1] * raw_h, coords[2] * raw_w, coords[3] * raw_h]
            elif all(0 <= value <= 1000 for value in coords) and (
                max(coords[0], coords[2]) > raw_w or max(coords[1], coords[3]) > raw_h
            ):
                raw_box = [
                    coords[0] / 1000 * raw_w,
                    coords[1] / 1000 * raw_h,
                    coords[2] / 1000 * raw_w,
                    coords[3] / 1000 * raw_h,
                ]
            else:
                raw_box = coords

            bboxes.append(self._raw_box_to_normalized(raw_box, raw_w=raw_w, raw_h=raw_h))
            return "<bbox>"

        converted_context = _COORD_RE.sub(replace, context)
        if not bboxes and fallback_box is not None:
            converted_context = inject_region_token(converted_context, "<bbox>")
            bboxes.append(self._raw_box_to_normalized(fallback_box, raw_w=raw_w, raw_h=raw_h))
        if not bboxes:
            return converted_context, None
        return converted_context, [torch.tensor(bboxes, device=self.device, dtype=self.dtype)]

    def _decode(self, output_ids):
        output_ids = output_ids[0][output_ids[0] != self.IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        text_output = text_output.split("ASSISTANT: ")[-1]
        text_output = re.sub(r"<.*?>", "", text_output)
        text_output = text_output.replace("[SEG]", "")
        return " ".join(text_output.split()).strip("'").strip()

    def _generate_one(self, context, gen_kwargs, visual, doc=None, task_name=None):
        if not visual:
            raise ValueError("GLaMM requires an image input.")
        image = visual[0].convert("RGB")
        fallback_box = get_doc_region_box_xyxy(doc, image, task_name=task_name)
        context, bboxes = self._prepare_region_prompt_and_bboxes(context, image, fallback_box=fallback_box)
        prompt = self._build_prompt(context)
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)
        max_new_tokens = get_task_max_new_tokens(gen_kwargs, task_name, 512)
        needs_predicted_mask = self.generate_masks or is_bbox_prediction_task(task_name)
        pred_masks = None

        with torch.inference_mode():
            if needs_predicted_mask:
                global_image, grounding_image, resize_list, original_size_list = self._prepare_images(image)
                output_ids, pred_masks = self.model.evaluate(
                    global_image,
                    grounding_image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_tokens_new=max_new_tokens,
                    bboxes=bboxes,
                )
            else:
                global_image = self._prepare_global_image(image)
                generation_kwargs = {
                    "images": global_image,
                    "input_ids": input_ids,
                    "bboxes": bboxes,
                    "max_new_tokens": max_new_tokens,
                    "num_beams": gen_kwargs.get("num_beams", 1),
                    "do_sample": gen_kwargs.get("temperature", 0) > 0,
                }
                if generation_kwargs["do_sample"]:
                    generation_kwargs["temperature"] = gen_kwargs.get("temperature", 1.0)
                    if gen_kwargs.get("top_p") is not None:
                        generation_kwargs["top_p"] = gen_kwargs["top_p"]
                output_ids = self.model.generate(**generation_kwargs)
        output = self._decode(output_ids)
        if is_bbox_prediction_task(task_name):
            predicted_box = _mask_to_normalized_bbox(pred_masks)
            if predicted_box is not None:
                return predicted_box
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        for term in until:
            if term and term in output:
                output = output.split(term)[0]
        return normalize_text_only_output(output.strip(), context, task_name)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=1, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            gen_kwargs = dict(all_gen_kwargs[0])
            doc = self.task_dict[task[0]][split[0]][doc_id[0]]
            visual = doc_to_visual[0](doc)
            output = self._generate_one(contexts[0], gen_kwargs, visual, doc=doc, task_name=task[0])
            res.append(output)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), output)
            pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("GLaMM adapter does not implement multi-round generation.")
