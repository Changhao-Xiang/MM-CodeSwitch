import re
from typing import Iterable, List, Optional, Tuple

KOSMOS_GROUNDING_TOKEN = "<grounding>"
KOSMOS_PHRASE_START = "<phrase>"
KOSMOS_PHRASE_END = "</phrase>"
KOSMOS_OBJECT_START = "<object>"
KOSMOS_OBJECT_END = "</object>"
KOSMOS_OBJECT_DELIMITER = "<delimiter_of_multi_objects/>"
KOSMOS_DEFAULT_NUM_BINS = 32


def kosmos_patch_index_token(index: int) -> str:
    return f"<patch_index_{index:04d}>"


def kosmos_base_tokens() -> List[str]:
    return [
        KOSMOS_GROUNDING_TOKEN,
        KOSMOS_PHRASE_START,
        KOSMOS_PHRASE_END,
        KOSMOS_OBJECT_START,
        KOSMOS_OBJECT_END,
        KOSMOS_OBJECT_DELIMITER,
    ]


def kosmos_coordinate_tokens(num_bins: int = KOSMOS_DEFAULT_NUM_BINS) -> List[str]:
    return [kosmos_patch_index_token(i) for i in range(num_bins * num_bins)]


def kosmos_tokens(num_bins: int = KOSMOS_DEFAULT_NUM_BINS) -> List[str]:
    return kosmos_base_tokens() + kosmos_coordinate_tokens(num_bins)


def coordinate_to_patch_index(x: float, y: float, width: int, height: int, num_bins: int) -> int:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")
    x_bin = max(0, min(num_bins - 1, int(x / width * num_bins)))
    y_bin = max(0, min(num_bins - 1, int(y / height * num_bins)))
    return y_bin * num_bins + x_bin


def bbox_to_kosmos_tokens(
    bbox: Iterable[float],
    image_size: Tuple[int, int],
    num_bins: int = KOSMOS_DEFAULT_NUM_BINS,
) -> Tuple[str, str]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    top_left = coordinate_to_patch_index(x1, y1, width, height, num_bins)
    bottom_right = coordinate_to_patch_index(x2, y2, width, height, num_bins)
    return kosmos_patch_index_token(top_left), kosmos_patch_index_token(bottom_right)


def normalized_bbox_to_kosmos_tokens(
    bbox: Iterable[float],
    num_bins: int = KOSMOS_DEFAULT_NUM_BINS,
) -> Tuple[str, str]:
    """Convert a normalized ``xyxy`` box to the two KOSMOS location tokens."""
    values = tuple(float(value) for value in bbox)
    if len(values) != 4:
        raise ValueError(f"Expected a 4-value xyxy box, got {values}")
    x1, y1, x2, y2 = values
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Normalized bbox values must be in [0, 1], got {values}")
    if x1 > x2 or y1 > y2:
        raise ValueError(f"Normalized bbox must satisfy x1 <= x2 and y1 <= y2, got {values}")
    return bbox_to_kosmos_tokens(values, image_size=(1, 1), num_bins=num_bins)


def format_kosmos_object(
    bbox: Iterable[float],
    image_size: Tuple[int, int],
    num_bins: int = KOSMOS_DEFAULT_NUM_BINS,
) -> str:
    loc_1, loc_2 = bbox_to_kosmos_tokens(bbox, image_size, num_bins)
    return f"{KOSMOS_OBJECT_START}{loc_1}{loc_2}{KOSMOS_OBJECT_END}"


def format_normalized_kosmos_grounding(
    phrase: str,
    bbox: Iterable[float],
    num_bins: int = KOSMOS_DEFAULT_NUM_BINS,
) -> str:
    """Format a referring-expression target using canonical KOSMOS-2 markup."""
    phrase = str(phrase).strip()
    if not phrase:
        raise ValueError("Grounding phrase must not be empty")
    loc_1, loc_2 = normalized_bbox_to_kosmos_tokens(bbox, num_bins=num_bins)
    return (
        f"{KOSMOS_GROUNDING_TOKEN}{KOSMOS_PHRASE_START}{phrase}{KOSMOS_PHRASE_END}"
        f"{KOSMOS_OBJECT_START}{loc_1}{loc_2}{KOSMOS_OBJECT_END}"
    )


_PATCH_INDEX_RE = re.compile(r"<patch_index_(\d{4})>")
_PHRASE_OBJECT_RE = re.compile(
    rf"{re.escape(KOSMOS_PHRASE_START)}(?P<phrase>.*?){re.escape(KOSMOS_PHRASE_END)}"
    rf"\s*{re.escape(KOSMOS_OBJECT_START)}(?P<object>.*?){re.escape(KOSMOS_OBJECT_END)}",
    flags=re.DOTALL,
)


def patch_index_to_center(index: int, num_bins: int = KOSMOS_DEFAULT_NUM_BINS) -> Tuple[float, float]:
    x_bin = index % num_bins
    y_bin = index // num_bins
    return (x_bin + 0.5) / num_bins, (y_bin + 0.5) / num_bins


def parse_kosmos_object(
    object_text: str,
    num_bins: int = KOSMOS_DEFAULT_NUM_BINS,
) -> List[Tuple[float, float, float, float]]:
    indices = [int(match.group(1)) for match in _PATCH_INDEX_RE.finditer(object_text)]
    boxes = []
    for i in range(0, len(indices) - 1, 2):
        x1, y1 = patch_index_to_center(indices[i], num_bins)
        x2, y2 = patch_index_to_center(indices[i + 1], num_bins)
        boxes.append((x1, y1, x2, y2))
    return boxes


def post_process_kosmos_generation(
    text: str,
    mode: Optional[str] = None,
    num_bins: int = KOSMOS_DEFAULT_NUM_BINS,
    coord_scale: int = 1000,
) -> str:
    if mode is None or mode == "raw":
        return text
    if mode not in {"strip", "text_bbox"}:
        raise ValueError(f"Unsupported KOSMOS postprocess mode: {mode}")

    def replace(match: re.Match) -> str:
        phrase = match.group("phrase")
        if mode == "strip":
            return phrase

        boxes = parse_kosmos_object(match.group("object"), num_bins)
        if not boxes:
            return phrase
        tags = []
        for x1, y1, x2, y2 in boxes:
            coords = (
                int(round(x1 * coord_scale)),
                int(round(y1 * coord_scale)),
                int(round(x2 * coord_scale)),
                int(round(y2 * coord_scale)),
            )
            tags.append(f"<box>({coords[0]},{coords[1]},{coords[2]},{coords[3]})</box>")
        return phrase + "".join(tags)

    text = _PHRASE_OBJECT_RE.sub(replace, text)
    if mode == "strip":
        text = _PATCH_INDEX_RE.sub("", text)
        for token in kosmos_base_tokens():
            text = text.replace(token, "")
    return text
