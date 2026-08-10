# MMCS: Multi-Modal Code-Switching

This repository contains the model, training, and evaluation code for MultiModal Code-Switching: Interleaving Visual Objects into Language for Explicit Object-Level Alignment [arXiv (Coming soon)](#).

Training consists of 773k-sample MMCS pretraining followed by 779k-sample LLaVA-NeXT LoRA supervised fine-tuning (SFT). Evaluation is provided through the bundled `lmms_eval` fork.



## Repository layout

- `common/`: shared command-line arguments and helpers.
- `model/`: model implementations, dynamic-resolution logic, code-switching, and projectors.
- `train/`: the training entry point, datasets, trainer, and data recipes.
- `scripts/train/`: two-stage training scripts for the released LLM backbones.
- `lmms_eval/`: vendored evaluation harness and MMCS model adapters.
- `scripts/eval/all.sh`: multi-benchmark evaluation entry point.
- `utils/`: data formatting and merging utilities.



## Installation

Python jobs should be launched from the repository root. The training environment was validated with Python 3.10.20, PyTorch 2.6.0 (CUDA 12.4), and Transformers 4.51.3:

```bash
conda create -n mmcs python=3.10.20 -y
conda activate mmcs
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e ./lmms_eval
```

FlashAttention is installed after PyTorch and the other dependencies because its build process imports PyTorch. We used version 2.7.4.post1; see the [official FlashAttention repository](https://github.com/Dao-AILab/flash-attention) for build requirements and prebuilt wheels compatible with your Python, PyTorch, and CUDA versions.

The key versions used for training are:

```text
Python        3.10
PyTorch       2.6.0+cu124
torchvision   0.21.0
torchaudio    2.6.0
Transformers  4.51.3
FlashAttention 2.7.4.post1
DeepSpeed     0.15.4
PEFT          0.14.0
NumPy         2.2.6
datasets      2.19.0
```

DeepSpeed, FlashAttention, and some evaluation tasks may require system packages or a CUDA toolkit compatible with the installed PyTorch version. Install task-specific optional dependencies only for the benchmarks you plan to run.



## Data

The released 773k pretraining annotations are hosted at [LockOnN/MMCS-Data](https://huggingface.co/datasets/LockOnN/MMCS-Data). This repository distributes generated annotations only; obtain the source images from their original providers and follow each dataset's license and terms.

Download the annotations from the repository root:

```bash
hf download LockOnN/MMCS-Data \
  --repo-type dataset \
  --include "annotations/*" \
  --local-dir data
```

Download the source images from the official sites or repositories below, then arrange them under `data/images` with the paths expected by the annotations:


| Source      | Official source                                                                                                                                                                      | Required layout under `data/images` |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| COCO 2014   | [COCO downloads](https://cocodataset.org/#download)                                                                                                                                  | `coco/train2014/*.jpg`              |
| Flickr30K   | [University of Illinois Flickr30K release](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html)                                                                          | `flickr30k/*.jpg`                   |
| GQA         | [Stanford GQA downloads](https://cs.stanford.edu/people/dorarad/gqa/download.html)                                                                                                   | `gqa/images/*.jpg`                  |
| Objects365  | [BAAI Objects365_2019 download](https://data.baai.ac.cn/datadetail/Objects365_2019)                                                                                                  | `objects365/train/*.jpg`            |
| Open Images | [CVDF Open Images repository](https://github.com/cvdfoundation/open-images-dataset) and [official](https://open-images-dataset.s3.amazonaws.com/tar/train_0.tar.gz) `train_0.tar.gz` | `openimages/train_0/*.jpg`          |
| SA-1B       | [Meta SA-1B download portal](https://ai.meta.com/datasets/segment-anything-downloads/) and [official repository](https://github.com/facebookresearch/segment-anything)               | `SA/*.jpg`                          |


For COCO, download the 2014 training images. For Flickr30K, download the image release after reviewing its Flickr usage terms. For GQA, download `Images.zip`. For Objects365, obtain the training images from the linked BAAI Objects365_2019 page and preserve the `obj365_train_*.jpg` filenames. Extract or link each source into the layout shown above.

The MMCS Open Images annotations reference 127,237 images whose official ImageIDs start with `0`. They are all contained in CVDF's single `train_0.tar.gz` image archive (approximately 46 GB). You may download and extract the archive directly, or use the downloader below to fetch only the referenced ImageIDs:

```bash
mkdir -p data/images/openimages/train_0

sed -n 's#.*"image": "openimages/train_0/\([^"]*\)\.jpg".*#train/\1#p' \
  data/annotations/openimages_127k_llava.json > data/openimages_mmcs_ids.txt

wget -O data/openimages_downloader.py \
  https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
pip install boto3 tqdm
python data/openimages_downloader.py data/openimages_mmcs_ids.txt \
  --download_folder=data/images/openimages/train_0 \
  --num_processes=16
```

For SA-1B, accept the SA-1B Research License on Meta's download portal and download `sa_000000.tar` through `sa_000010.tar` (11 archives in total).

The SFT stage uses the official 779k LLaVA-NeXT SFT release from [lmms-lab/LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data). Download the official raw-format JSON and image archives, then extract them in place:

```bash
hf download lmms-lab/LLaVA-NeXT-Data \
  --repo-type dataset \
  --include "llava_next_raw_format/*" \
  --local-dir data

for archive in data/llava_next_raw_format/*.tar.gz; do
  tar -xzf "$archive" -C data/llava_next_raw_format
done
```

A minimal final training-data layout should look like this (helper files and downloaded archives are omitted):

```text
data/
├── images/
│   ├── coco/
│   │   └── train2014/
│   │       └── *.jpg
│   ├── flickr30k/
│   │   └── *.jpg
│   ├── gqa/
│   │   └── images/
│   │       └── *.jpg
│   ├── objects365/
│   │   └── train/
│   │       └── *.jpg
│   ├── openimages/
│   │   └── train_0/
│   │       └── *.jpg
│   └── SA/
│       └── *.jpg
├── annotations/
│   ├── coco2014_75k_llava.json
│   ├── flickr29k_llava.json
│   ├── gqa_108k_llava.json
│   ├── objects365_343k_llava.json
│   ├── openimages_127k_llava.json
│   └── SA_91k_llava.json
└── llava_next_raw_format/
    ├── llava_next_raw_format_processed.json
    └── <extracted image directories>/
```

After preparation, the checked-in recipes use these paths:

- `train/recipe/mmcs.json`: images in `data/images` and annotations in `data/annotations`.
- `train/recipe/sft_779k.json`: the official 779k LLaVA-NeXT SFT data in `data/llava_next_raw_format`, including `llava_next_raw_format_processed.json`.

The default SFT recipe uses the standard, non-packed dataset format.

Recipe paths are relative to the repository root. Each entry has the form:

```json
{
  "name": "dataset_name",
  "image_folder": "data/images",
  "annotation": "data/annotations/train.json",
  "sample_ratio": 1.0
}
```



## Training

Choose a backbone and launch its two-stage training script from the repository root:


| Script                               | Vision encoder                      | Language model                        |
| ------------------------------------ | ----------------------------------- | ------------------------------------- |
| `scripts/train/siglip2-qwen25-3b.sh` | `google/siglip2-so400m-patch16-384` | `Qwen/Qwen2.5-3B-Instruct`            |
| `scripts/train/siglip2-qwen3-8b.sh`  | `google/siglip2-so400m-patch16-384` | `Qwen/Qwen3-8B`                       |
| `scripts/train/siglip2-llama3-8b.sh` | `google/siglip2-so400m-patch16-384` | `meta-llama/Meta-Llama-3-8B-Instruct` |


```bash
bash scripts/train/siglip2-qwen25-3b.sh
```

To use local model directories, set the path overrides when launching the selected script:

```bash
VISION_MODEL_PATH=/path/to/siglip2-so400m-patch16-384 \
LANGUAGE_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
bash scripts/train/siglip2-qwen25-3b.sh
```

`NUM_GPUS`, `MASTER_PORT`, `PRETRAIN_RECIPE`, `SFT_RECIPE`, `PRETRAIN_OUTPUT_DIR`, and `SFT_OUTPUT_DIR` can also be overridden through environment variables.

The script runs two stages in sequence:

1. MMCS pretraining with `train/recipe/mmcs.json`, a frozen SigLIP2 vision encoder, a frozen language model, and a trainable patch-merger projector.
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
