import re
import sys
from datetime import timedelta
from functools import partial
from typing import List, Optional, Tuple, Union

sys.modules.setdefault("deepspeed", None)

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import StoppingCriteria

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models._grounding_model_utils import (
    DEFAULT_LOCAL_CLIP_VISION_TOWER,
    disable_broken_deepspeed_import,
    ensure_repo_first,
    format_text_only_prompt,
    get_doc_region_box_xyxy,
    get_task_max_new_tokens,
    inject_region_token,
    is_bbox_prediction_task,
    normalize_text_only_output,
    route_clip_vision_tower,
)

DEFAULT_FERRET_REPO_PATH = "third_party/ferret"
DEFAULT_FERRET_CKPT_PATH = "checkpoints/ferret-7b-v1.3"
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"
VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000
_COORD_RE = re.compile(
    r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
)


class _KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            keyword_ids = tokenizer(keyword).input_ids
            if len(keyword_ids) > 1 and keyword_ids[0] == tokenizer.bos_token_id:
                keyword_ids = keyword_ids[1:]
            if keyword_ids:
                self.keyword_ids.append(torch.tensor(keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1."
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids.shape[1] >= keyword_id.shape[0] and torch.equal(
                output_ids[0, -keyword_id.shape[0] :], keyword_id
            ):
                return True

        offset = min(output_ids.shape[1] - self.start_len, 3)
        if offset <= 0:
            return False
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        return any(keyword in outputs for keyword in self.keywords)


def _generate_mask_for_feature(coor, raw_w, raw_h):
    coor_mask = np.zeros((raw_w, raw_h))
    x1, y1, x2, y2 = _clamp_box_xyxy(coor, raw_w=raw_w, raw_h=raw_h)
    coor_mask[x1 : x2 + 1, y1 : y2 + 1] = 1
    return torch.from_numpy(coor_mask)


def _clamp_box_xyxy(coor, raw_w, raw_h):
    x1, y1, x2, y2 = coor
    x1 = max(0, min(raw_w - 1, int(x1)))
    x2 = max(0, min(raw_w - 1, int(x2)))
    y1 = max(0, min(raw_h - 1, int(y1)))
    y2 = max(0, min(raw_h - 1, int(y2)))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [x1, y1, x2, y2]


def _box_to_text_vocab(raw_box, raw_w, raw_h):
    raw_box = _clamp_box_xyxy(raw_box, raw_w=raw_w, raw_h=raw_h)
    return [
        int(raw_box[0] / raw_w * VOCAB_IMAGE_W),
        int(raw_box[1] / raw_h * VOCAB_IMAGE_H),
        int(raw_box[2] / raw_w * VOCAB_IMAGE_W),
        int(raw_box[3] / raw_h * VOCAB_IMAGE_H),
    ]


def _prepare_region_prompt_and_masks(prompt, image, add_region_feature, fallback_box=None):
    if not add_region_feature:
        return prompt, None

    raw_w, raw_h = image.size
    masks = []

    def replace(match):
        coords = [float(match.group(i)) for i in range(1, 5)]
        if all(0 <= value <= 1 for value in coords):
            raw_box = [
                int(coords[0] * raw_w),
                int(coords[1] * raw_h),
                int(coords[2] * raw_w),
                int(coords[3] * raw_h),
            ]
            text_box = [
                int(coords[0] * VOCAB_IMAGE_W),
                int(coords[1] * VOCAB_IMAGE_H),
                int(coords[2] * VOCAB_IMAGE_W),
                int(coords[3] * VOCAB_IMAGE_H),
            ]
        else:
            raw_box = [int(value) for value in coords]
            text_box = _box_to_text_vocab(raw_box, raw_w=raw_w, raw_h=raw_h)
        masks.append(_generate_mask_for_feature(raw_box, raw_w=raw_w, raw_h=raw_h))
        return f"[{text_box[0]}, {text_box[1]}, {text_box[2]}, {text_box[3]}] {DEFAULT_REGION_FEA_TOKEN}"

    converted_prompt = _COORD_RE.sub(replace, prompt)
    if not masks and fallback_box is not None:
        raw_box = [int(value) for value in fallback_box]
        text_box = _box_to_text_vocab(raw_box, raw_w=raw_w, raw_h=raw_h)
        region_text = f"[{text_box[0]}, {text_box[1]}, {text_box[2]}, {text_box[3]}] {DEFAULT_REGION_FEA_TOKEN}"
        converted_prompt = inject_region_token(converted_prompt, region_text)
        masks.append(_generate_mask_for_feature(raw_box, raw_w=raw_w, raw_h=raw_h))
    return converted_prompt, masks or None


def _normalize_bbox_prediction(output, task_name):
    """Convert Ferret's native 0-1000 boxes to lmms_eval's 0-1 format."""
    if not is_bbox_prediction_task(task_name):
        return output

    match = _COORD_RE.search(output)
    if match is None:
        return output

    coords = [float(match.group(i)) for i in range(1, 5)]
    if any(abs(value) > 1 for value in coords):
        coords = [value / 1000 for value in coords]
    coords = [max(0.0, min(1.0, value)) for value in coords]
    return "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*coords)


@register_model("ferret")
class Ferret(lmms):
    def __init__(
        self,
        pretrained: str = DEFAULT_FERRET_CKPT_PATH,
        model_base: Optional[str] = None,
        ferret_repo_path: str = DEFAULT_FERRET_REPO_PATH,
        vision_tower_path: Optional[str] = DEFAULT_LOCAL_CLIP_VISION_TOWER,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        conv_mode: str = "ferret_v1",
        add_region_feature: bool = True,
        image_size: int = 336,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        if not torch.cuda.is_available():
            raise RuntimeError("Ferret inference requires CUDA because the official model loader moves CLIP to cuda.")

        disable_broken_deepspeed_import()
        route_clip_vision_tower(vision_tower_path)
        ensure_repo_first(ferret_repo_path)

        from ferret.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from ferret.conversation import SeparatorStyle, conv_templates
        from ferret.mm_utils import get_model_name_from_path, tokenizer_image_token
        from ferret.model.builder import load_pretrained_model

        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self._device = torch.device(f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else device)
        if torch.cuda.is_available():
            torch.cuda.set_device(self._device)
        self.device_map = str(self._device) if torch.cuda.is_available() else "cpu"

        model_name = get_model_name_from_path(pretrained)
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained, model_base, model_name, device_map=self.device_map
        )
        self._config = self._model.config
        self._model.eval()

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
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.batch_size_per_gpu = int(batch_size)
        self.conv_mode = conv_mode
        self.add_region_feature = add_region_feature
        self.image_size = image_size
        self.use_cache = use_cache

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
        return self._max_length

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
        raise NotImplementedError("Ferret lmms_eval adapter currently supports generate_until only.")

    def _flatten(self, items):
        if not items or any(item is None for item in items):
            return []
        return [entry for item in items for entry in item]

    def _prepare_image(self, image: Image.Image):
        image = image.convert("RGB")
        return self._image_processor.preprocess(
            image,
            return_tensors="pt",
            do_resize=True,
            do_center_crop=False,
            size=[self.image_size, self.image_size],
        )["pixel_values"][0]

    def _generate_one(self, context, gen_kwargs, visual, doc=None, task_name=None):
        image = visual[0].convert("RGB") if visual else None
        text_only = not is_bbox_prediction_task(task_name)
        task_name_lower = str(task_name or "").lower()
        should_format_prompt = text_only and not task_name_lower.startswith(("mmbench", "ocrbench"))
        prompt_context = format_text_only_prompt(context, task_name) if should_format_prompt else context
        region_masks = None
        if image is not None:
            fallback_box = get_doc_region_box_xyxy(doc, image, task_name=task_name)
            prompt_context, region_masks = _prepare_region_prompt_and_masks(
                prompt_context, image, self.add_region_feature, fallback_box=fallback_box
            )
            if self.config.mm_use_im_start_end:
                question = (
                    self.DEFAULT_IM_START_TOKEN
                    + self.DEFAULT_IMAGE_TOKEN
                    + self.DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt_context
                )
            else:
                question = self.DEFAULT_IMAGE_TOKEN + "\n" + prompt_context
        else:
            question = prompt_context

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )

        image_tensor = None
        if image is not None:
            image_tensor = self._prepare_image(image).unsqueeze(0).to(device=self.device, dtype=torch.float16)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        until = gen_kwargs.get("until", [stop_str])
        if isinstance(until, str):
            until = [until]
        keywords = list(dict.fromkeys([stop_str] + until))
        stopping_criteria = _KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        max_new_tokens = get_task_max_new_tokens(gen_kwargs, task_name, 1024)
        temperature = gen_kwargs.get("temperature", 0)
        top_p = gen_kwargs.get("top_p", None)
        num_beams = gen_kwargs.get("num_beams", 1)

        if region_masks is not None:
            region_masks = [[mask.to(device=self.device, dtype=torch.float16) for mask in region_masks]]

        with torch.inference_mode():
            if region_masks is not None:
                self.model.orig_forward = self.model.forward
            try:
                if region_masks is not None:
                    self.model.forward = partial(self.model.orig_forward, region_masks=region_masks)
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                )
            finally:
                if region_masks is not None:
                    self.model.forward = self.model.orig_forward

        output = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)[0].strip()
        for term in keywords:
            if term and term in output:
                output = output.split(term)[0]
        output = _normalize_bbox_prediction(output.strip(), task_name)
        return normalize_text_only_output(output, context, task_name)

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
        raise NotImplementedError("Ferret adapter does not implement multi-round generation.")
