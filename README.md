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

The released dataset for pretraining is hosted at [MMCS-Data](https://huggingface.co/datasets/LockOnN/MMCS-Data).

Download the datasets under `data/`, or update the paths in the checked-in recipes:

- `train/recipe/mmcs.json`: 773k MMCS pretraining data. Images are read from `data/images`, and annotations from `data/segment`.
- `train/recipe/sft_779k.json`: 779k LLaVA-NeXT SFT data from [lmms-lab/LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data), stored under `data/llava_next_raw_format`.

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
