# MMCS: Multi-Modal Code-Switching

This repository contains the model, training, and evaluation code for MMCS(Multimodal Code-Switching). MMCS extends a LLaVA-style vision-language model with interleaved object visual tokens.

The released configuration uses SigLIP2 as the vision encoder and Qwen2.5-3B-Instruct as the language model. Training consists of 773k-sample MMCS pretraining followed by 779k-sample LLaVA-NeXT LoRA supervised fine-tuning (SFT). Evaluation is provided through the bundled `lmms_eval` fork.

## Repository layout

- `common/`: shared command-line arguments and helpers.
- `model/`: LLaVA-NeXT model implementations, dynamic-resolution logic, MMCS code-switching, and projectors.
- `train/`: the training entry point, datasets, trainer, and data recipes.
- `scripts/train/mmcs.sh`: the two-stage training pipeline.
- `lmms_eval/`: vendored evaluation harness and MMCS model adapters.
- `scripts/eval/all.sh`: multi-benchmark evaluation entry point.
- `utils/`: data formatting and merging utilities.

## Installation

Python jobs should be launched from the repository root. The research environment uses Python 3.10 and CUDA-enabled PyTorch:

```bash
conda create -n mmcs python=3.10 -y
conda activate mmcs
pip install -r requirements.txt
pip install -e ./lmms_eval
```

`flash-attn`, DeepSpeed, and some evaluation tasks may require a compiler and a CUDA toolkit compatible with the installed PyTorch version. Install task-specific optional dependencies only for the benchmarks you plan to run.

## Data

The released 773k pretraining annotations are hosted at [LockOnN/MMCS-Data](https://huggingface.co/datasets/LockOnN/MMCS-Data). This repository distributes generated annotations only; obtain the source images from their original providers and follow each dataset's license and terms.

Download the annotations from the repository root:

```bash
hf download LockOnN/MMCS-Data \
  --repo-type dataset \
  --include "annotations/*" \
  --local-dir data/mmcs_download

ln -s mmcs_download/annotations data/segment
```

If `data/segment` already exists, replace the symbolic-link command with the equivalent copy or path update for your setup.

Download the source images from the official sites or repositories below, then arrange them under `data/images` with the paths expected by the annotations:

| Source | Official source | Required layout under `data/images` |
| --- | --- | --- |
| COCO 2014 | [COCO downloads](https://cocodataset.org/#download) | `coco/train2014/*.jpg` |
| Flickr30K | [University of Illinois Flickr30K release](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html) | `flickr30k/*.jpg` |
| GQA | [Stanford GQA downloads](https://cs.stanford.edu/people/dorarad/gqa/download.html) | `gqa/images/*.jpg` |
| Objects365 | [BAAI Objects365_2019 download](https://data.baai.ac.cn/datadetail/Objects365_2019) | `objects365/train/*.jpg` |
| Open Images | [CVDF Open Images repository](https://github.com/cvdfoundation/open-images-dataset) and [official `train_0.tar.gz`](https://open-images-dataset.s3.amazonaws.com/tar/train_0.tar.gz) | `openimages/train_0/*.jpg` |
| SA-1B | [Meta SA-1B download portal](https://ai.meta.com/datasets/segment-anything-downloads/) and [official repository](https://github.com/facebookresearch/segment-anything) | `SA/*.jpg` |

For COCO, download the 2014 training images. For Flickr30K, download the image release after reviewing its Flickr usage terms. For GQA, download `Images.zip`. For Objects365, obtain the training images from the linked BAAI Objects365_2019 page and preserve the `obj365_train_*.jpg` filenames. Extract or link each source into the layout shown above.

The MMCS Open Images annotations reference 127,237 images whose official ImageIDs start with `0`. They are all contained in CVDF's single `train_0.tar.gz` image archive (approximately 46 GB). The similarly named `open-images-dataset-train0.tsv` is a separate one-million-row manifest for transferring original image URLs and is not the `train_0` image archive. You may download and extract the archive directly, or use the official downloader below to fetch only the referenced ImageIDs:

```bash
mkdir -p data/images/openimages/train_0

sed -n 's#.*"image": "openimages/train_0/\([^"]*\)\.jpg".*#train/\1#p' \
  data/segment/openimages_127k_llava.json > data/openimages_mmcs_ids.txt

wget -O data/openimages_downloader.py \
  https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
pip install boto3 tqdm
python data/openimages_downloader.py data/openimages_mmcs_ids.txt \
  --download_folder=data/images/openimages/train_0 \
  --num_processes=16
```

For SA-1B, accept the SA-1B Research License on Meta's download portal and download `sa_000000.tar` through `sa_000010.tar` (11 archives in total). The released annotations reference 91,107 unique images from these archives, ranging from `sa_2.jpg` through `sa_111876.jpg`. Only the referenced `.jpg` files are needed for MMCS; place them directly in `data/images/SA`.

Only files referenced by the annotations are required. After preparing the directories, verify that every annotation resolves under `data/images` without loading the multi-gigabyte JSON files into memory:

```bash
python - <<'PY'
import re
from pathlib import Path

image_root = Path("data/images")
annotation_root = Path("data/segment")
image_pattern = re.compile(r'^\s*"image": "([^"]+)"')
missing_count = 0
missing_examples = []

for annotation in sorted(annotation_root.glob("*_llava.json")):
    with annotation.open(encoding="utf-8") as stream:
        for line in stream:
            match = image_pattern.match(line)
            if match and not (image_root / match.group(1)).is_file():
                missing_count += 1
                if len(missing_examples) < 20:
                    missing_examples.append((annotation.name, match.group(1)))

print(f"missing images: {missing_count}")
for annotation, image in missing_examples:
    print(f"{annotation}: {image}")
raise SystemExit(1 if missing_count else 0)
PY
```

The second stage uses the official 779k LLaVA-NeXT SFT release from [lmms-lab/LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data). Download the official raw-format JSON and image archives, then extract them in place:

```bash
hf download lmms-lab/LLaVA-NeXT-Data \
  --repo-type dataset \
  --include "llava_next_raw_format/*" \
  --local-dir data

for archive in data/llava_next_raw_format/*.tar.gz; do
  tar -xzf "$archive" -C data/llava_next_raw_format
done
```

After preparation, the checked-in recipes use these paths:

- `train/recipe/mmcs.json`: images in `data/images` and annotations in `data/segment`.
- `train/recipe/sft_779k.json`: the official 779k LLaVA-NeXT SFT data in `data/llava_next_raw_format`, including `llava_next_raw_format_processed.json`.

The default SFT recipe uses the standard, non-packed dataset format.

Recipe paths are relative to the repository root. Each entry has the form:

```json
{
  "name": "dataset_name",
  "image_folder": "data/images",
  "annotation": "data/segment/train.json",
  "sample_ratio": 1.0
}
```

## Training

Launch training from the repository root:

```bash
bash scripts/train/mmcs.sh
```

The script runs two stages in sequence:

1. MMCS pretraining with `train/recipe/mmcs.json`, a frozen SigLIP2 vision encoder, a frozen Qwen2.5-3B-Instruct language model, and a trainable patch-merger projector.
2. LoRA SFT with `train/recipe/sft_779k.json`.

## Evaluation

Run the checked-in benchmark suite with:

```bash
MODEL_CKPT=checkpoints/siglip2-qwen25-next-3B-sft-779k-lora \
  bash scripts/eval/all.sh
```

Override `MODEL_CKPT`, `TASKS`, `NUM_PROCESSES`, or `OUTPUT_PATH` as needed; additional arguments are forwarded to `lmms_eval`.

## Acknowledgements

The model architecture was implemented with reference to [InternVL](https://github.com/OpenGVLab/InternVL). We thank the OpenGVLab and InternVL contributors for open-sourcing their work.
The evaluation code in this repository is based on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank the LMMS-Lab contributors for developing and open-sourcing the project.
