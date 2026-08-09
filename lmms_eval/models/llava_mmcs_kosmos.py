import copy
import re
from typing import List, Optional

from common.kosmos_tokens import (
    KOSMOS_DEFAULT_NUM_BINS,
    KOSMOS_OBJECT_END,
    KOSMOS_OBJECT_START,
    parse_kosmos_object,
    post_process_kosmos_generation,
)
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models._grounding_model_utils import get_task_max_new_tokens
from lmms_eval.models.llava_mmcs import Llava_mmcs

KOSMOS_GROUNDING_INSTRUCTION = (
    "Locate the image region described by the text. Answer only with KOSMOS-2 grounding markup."
)

_KOSMOS_OBJECT_RE = re.compile(
    rf"{re.escape(KOSMOS_OBJECT_START)}(?P<object>.*?){re.escape(KOSMOS_OBJECT_END)}",
    flags=re.DOTALL,
)
_NUMERIC_BOX_RE = re.compile(
    r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*" r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _normalize_box(box):
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return x1, y1, x2, y2


def _extract_kosmos_boxes(text: str, num_bins: int):
    boxes = []
    for match in _KOSMOS_OBJECT_RE.finditer(text):
        boxes.extend(parse_kosmos_object(match.group("object"), num_bins=num_bins))
    if not boxes:
        boxes.extend(parse_kosmos_object(text, num_bins=num_bins))
    return [_normalize_box(box) for box in boxes]


def _format_bbox_list(box) -> str:
    return "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*box)


def _extract_numeric_box(text: str):
    match = _NUMERIC_BOX_RE.search(text)
    if match is None:
        return None
    box = [float(match.group(i)) for i in range(1, 5)]
    if any(abs(value) > 1 for value in box):
        box = [value / 1000 for value in box]
    return _normalize_box(box)


def post_process_kosmos_eval_output(
    text: str,
    mode: Optional[str],
    num_bins: int = KOSMOS_DEFAULT_NUM_BINS,
) -> str:
    if mode is None or mode == "raw":
        return text

    if mode in {"strip", "text_bbox"}:
        return post_process_kosmos_generation(text, mode=mode, num_bins=num_bins)

    if mode in {"bbox", "bbox_list", "grounding", "normalized_bbox"}:
        boxes = _extract_kosmos_boxes(text, num_bins=num_bins)
        if boxes:
            return _format_bbox_list(boxes[0])
        numeric_box = _extract_numeric_box(text)
        if numeric_box is not None:
            return _format_bbox_list(numeric_box)
        return post_process_kosmos_generation(text, mode="strip", num_bins=num_bins)

    raise ValueError(f"Unsupported KOSMOS eval postprocess mode: {mode}")


@register_model("llava_mmcs_kosmos")
class Llava_mmcs_kosmos(Llava_mmcs):
    """
    LLaVA-MMCS evaluation wrapper for KOSMOS-2 style coordinate-token outputs.

    It keeps the original llava_mmcs model path untouched while adding:
    - decoding with coordinate tokens preserved
    - optional grounding-prompt rewrite for RefCOCO/ScreenSpot-style tasks
    - normalized bbox postprocess compatible with existing grounding metrics
    """

    def __init__(
        self,
        *args,
        kosmos_postprocess: Optional[str] = "auto",
        kosmos_rewrite_grounding_prompt: bool = True,
        kosmos_grounding_tasks: str = "refcoco,refcoco+,refcocog,screenspot",
        **kwargs,
    ) -> None:
        self.kosmos_eval_postprocess = kosmos_postprocess
        self.kosmos_rewrite_grounding_prompt = _as_bool(kosmos_rewrite_grounding_prompt)
        self.kosmos_grounding_tasks = tuple(_split_csv(kosmos_grounding_tasks))
        super().__init__(*args, kosmos_postprocess=None, **kwargs)
        self._config.enable_coordinate_tokens = True
        if not hasattr(self._config, "coordinate_token_bins"):
            self._config.coordinate_token_bins = KOSMOS_DEFAULT_NUM_BINS

    def _is_grounding_task(self, task: str) -> bool:
        task = str(task).lower()
        return any(task.startswith(prefix.lower()) for prefix in self.kosmos_grounding_tasks)

    def _extract_grounding_text(self, context: str) -> str:
        lower_context = context.lower()
        markers = [
            "the region this sentence describes:",
            "the region that corresponds to the command:",
            "the command:",
            "sentence describes:",
            "command:",
        ]
        for marker in markers:
            marker_index = lower_context.rfind(marker)
            if marker_index >= 0:
                return context[marker_index + len(marker) :].strip()
        return context.strip()

    def _rewrite_grounding_context(self, context: str, task: str) -> str:
        if not self.kosmos_rewrite_grounding_prompt or not self._is_grounding_task(task):
            return context

        grounding_text = self._extract_grounding_text(context)
        return f"{KOSMOS_GROUNDING_INSTRUCTION}\nText: {grounding_text}"

    def _rewrite_request(self, request: Instance) -> Instance:
        context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
        rewritten_context = self._rewrite_grounding_context(context, task)
        rewritten_gen_kwargs = copy.deepcopy(gen_kwargs)
        if self._is_grounding_task(task):
            # Grounding answers need only a short markup/coordinate sequence.
            # Bounding this prevents malformed responses from running to the
            # generic 1024-token default for every RefCOCO expression.
            rewritten_gen_kwargs["max_new_tokens"] = min(rewritten_gen_kwargs.get("max_new_tokens", 128), 128)
        rewritten_gen_kwargs["max_new_tokens"] = get_task_max_new_tokens(rewritten_gen_kwargs, task, 1024)
        if rewritten_context == context and rewritten_gen_kwargs == gen_kwargs:
            return request

        rewritten_request = copy.copy(request)
        rewritten_request.arguments = (rewritten_context, rewritten_gen_kwargs, doc_to_visual, doc_id, task, split)
        return rewritten_request

    def _postprocess_mode_for_task(self, task: str):
        if self.kosmos_eval_postprocess != "auto":
            return self.kosmos_eval_postprocess
        return "bbox_list" if self._is_grounding_task(task) else "strip"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        rewritten_requests = [self._rewrite_request(request) for request in requests]
        outputs = super().generate_until(rewritten_requests)
        num_bins = getattr(self._config, "coordinate_token_bins", KOSMOS_DEFAULT_NUM_BINS)
        return [
            post_process_kosmos_eval_output(
                output,
                mode=self._postprocess_mode_for_task(request.args[4]),
                num_bins=num_bins,
            )
            for output, request in zip(outputs, requests)
        ]
