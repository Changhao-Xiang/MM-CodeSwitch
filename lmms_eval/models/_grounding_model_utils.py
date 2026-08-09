import os
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_CLIP_VISION_TOWER = "openai/clip-vit-large-patch14-336"
DEFAULT_LOCAL_CLIP_VISION_TOWER = DEFAULT_CLIP_VISION_TOWER
SHORT_ANSWER_GENERATION_LIMITS = {
    "mmbench": 32,
    "mmstar": 64,
    "mmmu": 64,
    "mmvet": 512,
    "textvqa": 64,
}
MINIMUM_GENERATION_TOKENS = {
    "chartqa": 64,
    "gqa": 64,
    "mme": 32,
}

SHORT_ANSWER_TASK_PREFIXES = (
    "chartqa",
    "gqa",
    "ocrbench",
    "realworldqa",
    "textvqa",
)

_COORDINATE_BOX_RE = re.compile(
    r"[\[\(]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*[\]\)]"
)
_CHOICE_LINE_RE = re.compile(r"(?m)^\s*(?:\(([A-Z])\)|([A-Z])[.):])\s+(.*?)\s*$")
_INLINE_CHOICE_MARKER_RE = re.compile(r"(?:^|,\s*)([A-Z]):\s*")
_STYLE_PREFIX_RE = re.compile(
    r"^(?:in (?:a )?nutshell|in (?:one|1) phrase|in short|short description|brief observation)\s*[:,.-]?\s*",
    re.IGNORECASE,
)


def format_text_only_prompt(context, task_name=None, prohibit_grounding=True):
    """Keep grounding models from answering ordinary VQA with regions or masks."""
    context = context.rstrip()
    task_name = str(task_name or "").lower()
    context_lower = context.lower()

    instructions = []
    if prohibit_grounding:
        instructions.append("Do not include segmentation masks, region tokens, or bounding-box coordinates.")
    if "answer yes or no" in context_lower or "yes or no" in context_lower:
        response_format = "Respond with exactly Yes or No."
    elif _parse_multiple_choices(context):
        response_format = "Respond with only the letter of the correct option."
    elif task_name.startswith(SHORT_ANSWER_TASK_PREFIXES) or any(
        cue in context_lower
        for cue in (
            "single word",
            "single phrase",
            "single word or phrase",
        )
    ):
        response_format = "Respond with only the short answer, without explanation."
    else:
        response_format = "Answer the question directly."

    instructions.extend((response_format, "Answer:"))
    return f"{context}\n{' '.join(instructions)}"


def _strip_grounding_markup(output):
    output = _COORDINATE_BOX_RE.sub(" ", output)
    output = re.sub(r"\s*[\[\(]\s*-?\d+(?:\s*,\s*-?\d+){0,3}\s*$", " ", output)
    output = re.sub(r"<[^>]*>|\[SEG\]|<region_fea>", " ", output, flags=re.IGNORECASE)
    output = _STYLE_PREFIX_RE.sub("", output.strip())
    output = re.sub(r"\s+", " ", output)
    return output.strip(" \t\r\n'\".,;:")


def _parse_multiple_choices(context):
    choices = [(parenthesized or bare, text) for parenthesized, bare, text in _CHOICE_LINE_RE.findall(context)]
    if choices:
        return choices

    for line in context.splitlines():
        if not line.strip().lower().startswith("options:"):
            continue
        choices_text = line.split(":", 1)[1].strip()
        markers = list(_INLINE_CHOICE_MARKER_RE.finditer(choices_text))
        if len(markers) < 2:
            continue
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(choices_text)
            text = choices_text[marker.end() : end].strip(" ,")
            choices.append((marker.group(1), text))
        return choices
    return choices


def _extract_multiple_choice(output, context):
    choices = _parse_multiple_choices(context)
    if not choices:
        return None

    raw_choice = re.match(r"^\s*\(?([A-Z])\)?(?:\s*[.)]|\s*(?=\[)|\s*$)", output)
    if raw_choice and any(letter == raw_choice.group(1) for letter, _ in choices):
        return raw_choice.group(1)

    cleaned = _strip_grounding_markup(output)
    if re.fullmatch(r"\(?([A-Z])\)?", cleaned):
        return cleaned.strip("()")

    normalized_output = cleaned.casefold()
    matches = []
    for letter, choice in choices:
        choice = choice.strip().strip(" \t\r\n'\".,;:")
        if not choice:
            continue
        if len(choice) == 1 and normalized_output != choice.casefold():
            continue
        match = re.search(rf"(?<!\w){re.escape(choice.casefold())}(?!\w)", normalized_output)
        if match:
            matches.append((match.start(), -len(choice), letter))
    return min(matches)[2] if matches else None


def _extract_short_answer(output, context):
    cleaned = _strip_grounding_markup(output)
    if not cleaned:
        return cleaned

    context_lower = context.lower()
    leading_yes_no = re.match(r"^(yes|no)\b", cleaned, re.IGNORECASE)
    if leading_yes_no:
        return leading_yes_no.group(1)

    yes_no = re.search(r"\b(yes|no)\b", cleaned, re.IGNORECASE)
    if (
        "yes or no" in context_lower
        or re.match(r"^(?:is|are|do|does|did|can|could|will|would|has|have)\b", context_lower)
    ) and yes_no:
        return yes_no.group(1)

    quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", cleaned)
    if quoted and any(
        cue in context_lower
        for cue in ("what does", "what is written", "what word", "what title", "what is the name", "who edited")
    ):
        return quoted[-1].strip(" \t\r\n'\".,;:")

    if any(cue in context_lower for cue in ("what time", "the time")):
        time_match = re.search(r"\b\d{1,2}:\d{2}\b", cleaned)
        if time_match:
            return time_match.group(0)

    if any(
        cue in context_lower
        for cue in ("how many", "how much", "how long", "what year", "the year", "what number", "percent")
    ):
        number_matches = re.findall(
            r"(?:\$\s*)?\d+(?::\d+|\.\d+)?\s*(?:%|percent|years?|dollars?)?", cleaned, re.IGNORECASE
        )
        if number_matches:
            return number_matches[-1].strip()

    alternatives = re.search(r"\b([\w-]+)\s+or\s+(?:maybe\s+)?([\w-]+)\??\s*$", context, re.IGNORECASE)
    if alternatives:
        for candidate in alternatives.groups():
            if re.search(rf"\b{re.escape(candidate)}\b", cleaned, re.IGNORECASE):
                return candidate

    complement_patterns = (
        r"\b(?:aged for|edited by|spells?(?: out)?|reads?|says?)\s+(.+)$",
        r"\b(?:answer|brand|color|material|side|name|title|word|number|year|time)\s+(?:is|are|was|were)\s+(.+)$",
        r"\b(?:is|are|was|were)\s+(.+)$",
    )
    for pattern in complement_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            answer = match.group(1).strip(" \t\r\n'\".,;:")
            if answer and len(answer.split()) <= 12:
                return answer

    if re.match(r"^who\b", context_lower):
        subject = re.match(r"^(?:a|an|the)\s+(.+?)(?:\s+(?:is|are|was|were)\b|[,.]|$)", cleaned, re.IGNORECASE)
        if subject:
            return subject.group(1).strip()

    return cleaned


def normalize_text_only_output(output, context, task_name=None):
    """Convert verbose grounding-model replies to the answer formats used by VQA tasks."""
    if is_bbox_prediction_task(task_name):
        return output

    multiple_choice = _extract_multiple_choice(output, context)
    if multiple_choice is not None:
        return multiple_choice

    task_name = str(task_name or "").lower()
    context_lower = context.lower()
    if (
        task_name.startswith(SHORT_ANSWER_TASK_PREFIXES)
        or "yes or no" in context_lower
        or any(cue in context_lower for cue in ("single word", "single phrase", "single word or phrase"))
    ):
        return _extract_short_answer(output, context)
    return _strip_grounding_markup(output)


def get_task_max_new_tokens(gen_kwargs, task_name, default):
    max_new_tokens = gen_kwargs.get("max_new_tokens", gen_kwargs.get("max_tokens_new", default))
    task_name = str(task_name or "").lower()
    if is_bbox_prediction_task(task_name):
        return min(max_new_tokens, 64)
    for task_prefix, minimum in MINIMUM_GENERATION_TOKENS.items():
        if task_name.startswith(task_prefix):
            max_new_tokens = max(max_new_tokens, minimum)
    for task_prefix, limit in SHORT_ANSWER_GENERATION_LIMITS.items():
        if task_name.startswith(task_prefix):
            return min(max_new_tokens, limit)
    return max_new_tokens


def is_bbox_prediction_task(task_name):
    if not task_name:
        return False
    task_name = task_name.lower()
    return "bbox_rec" in task_name or task_name.startswith("screenspot_rec")


def get_doc_region_box_xyxy(doc, image, task_name=None):
    if doc is None or image is None or is_bbox_prediction_task(task_name):
        return None
    if "bbox" not in doc:
        return None

    bbox = doc["bbox"]
    if bbox is None or len(bbox) != 4:
        return None

    raw_w, raw_h = image.size
    coords = [float(value) for value in bbox]
    task_name = (task_name or "").lower()

    if all(0 <= value <= 1 for value in coords):
        x1, y1, x2, y2 = coords
        return [x1 * raw_w, y1 * raw_h, x2 * raw_w, y2 * raw_h]

    if "refcoco" in task_name:
        x1, y1, width, height = coords
        return [x1, y1, x1 + width, y1 + height]

    return coords


def inject_region_token(context, token):
    if token in context:
        return context
    if "this region" in context:
        return context.replace("this region", f"this region {token}", 1)
    if "highlighted region" in context:
        return context.replace("highlighted region", f"highlighted region {token}", 1)
    return f"{context.rstrip()} {token}"


def disable_broken_deepspeed_import():
    # The project vlm env has deepspeed installed, but importing it can fail before inference starts.
    # Weight-only inference paths here do not need deepspeed.
    sys.modules.setdefault("deepspeed", None)


def resolve_path(path):
    return str(Path(os.path.expandvars(os.path.expanduser(path))).resolve())


def ensure_repo_first(repo_path):
    repo_path = resolve_path(repo_path)
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)
    return repo_path


def purge_conflicting_top_level_modules(repo_path, module_roots):
    repo_path = resolve_path(repo_path)
    for name, module in list(sys.modules.items()):
        root = name.split(".", 1)[0]
        if root not in module_roots:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None or not resolve_path(module_file).startswith(repo_path):
            del sys.modules[name]


def install_repo_namespace_packages(repo_path, package_names):
    repo_path = Path(resolve_path(repo_path))
    for package_name in package_names:
        package_path = repo_path / package_name
        if not package_path.exists():
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]
        module.__package__ = package_name
        sys.modules[package_name] = module


def allow_transformers_duplicate_registration():
    from transformers import AutoConfig, AutoModelForCausalLM

    if not hasattr(AutoConfig, "_lmms_eval_orig_register"):
        AutoConfig._lmms_eval_orig_register = AutoConfig.register

        def register_config(model_type, config, exist_ok=False):
            return AutoConfig._lmms_eval_orig_register(model_type, config, exist_ok=True)

        AutoConfig.register = register_config

    if not hasattr(AutoModelForCausalLM, "_lmms_eval_orig_register"):
        AutoModelForCausalLM._lmms_eval_orig_register = AutoModelForCausalLM.register

        def register_model(config_class, model_class, exist_ok=False):
            return AutoModelForCausalLM._lmms_eval_orig_register(config_class, model_class, exist_ok=True)

        AutoModelForCausalLM.register = register_model


def route_clip_vision_tower(vision_tower_path=None):
    if vision_tower_path is None:
        vision_tower_path = DEFAULT_LOCAL_CLIP_VISION_TOWER
    if not vision_tower_path:
        return

    vision_tower_path = resolve_path(vision_tower_path)
    if not Path(vision_tower_path).exists():
        return

    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

    def _patch(cls):
        if not hasattr(cls, "_lmms_eval_orig_from_pretrained"):
            cls._lmms_eval_orig_from_pretrained = cls.from_pretrained

        orig = cls._lmms_eval_orig_from_pretrained

        def from_pretrained(routed_cls, pretrained_model_name_or_path, *args, **kwargs):
            if pretrained_model_name_or_path == DEFAULT_CLIP_VISION_TOWER:
                pretrained_model_name_or_path = vision_tower_path
                kwargs.setdefault("local_files_only", True)
            return orig(pretrained_model_name_or_path, *args, **kwargs)

        cls.from_pretrained = classmethod(from_pretrained)

    for cls in (CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel):
        _patch(cls)


class _CompatRoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=0, **kwargs):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, feats, rois):
        from torchvision.ops import roi_align

        return roi_align(
            feats,
            rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=True,
        )


class _CompatBaseRoIExtractor(nn.Module):
    def __init__(self, roi_layer, out_channels, featmap_strides, init_cfg=None):
        super().__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.fp16_enabled = False

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop("type")
        if layer_type != "RoIAlign":
            raise ValueError(f"Unsupported lightweight mmcv op: {layer_type}")
        return nn.ModuleList([_CompatRoIAlign(spatial_scale=1 / s, **cfg) for s in featmap_strides])

    @property
    def num_inputs(self):
        return len(self.featmap_strides)


class _CompatConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if norm_cfg and norm_cfg.get("type") == "GN":
            self.gn = nn.GroupNorm(norm_cfg.get("num_groups", 32), out_channels)
        else:
            self.gn = None
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        return self.activate(x)


def _normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def install_lightweight_mmcv_compat():
    try:
        import mmcv  # noqa: F401
        import mmdet.models  # noqa: F401

        return
    except Exception:
        pass

    mmcv = types.ModuleType("mmcv")
    mmcv.__version__ = "1.4.0"
    mmcv.ops = types.ModuleType("mmcv.ops")
    mmcv.ops.RoIAlign = _CompatRoIAlign

    mmcv_cnn = types.ModuleType("mmcv.cnn")
    mmcv_cnn.ConvModule = _CompatConvModule
    mmcv_cnn.Linear = nn.Linear
    mmcv_cnn.normal_init = _normal_init

    mmcv_runner = types.ModuleType("mmcv.runner")
    mmcv_runner.BaseModule = nn.Module

    mmdet = types.ModuleType("mmdet")
    mmdet_models = types.ModuleType("mmdet.models")
    mmdet_models.BaseRoIExtractor = _CompatBaseRoIExtractor
    mmcv.cnn = mmcv_cnn
    mmcv.runner = mmcv_runner
    mmdet.models = mmdet_models

    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.ops"] = mmcv.ops
    sys.modules["mmcv.cnn"] = mmcv_cnn
    sys.modules["mmcv.runner"] = mmcv_runner
    sys.modules["mmdet"] = mmdet
    sys.modules["mmdet.models"] = mmdet_models
