import json
import os
import random

from tqdm import tqdm


def format_captions_to_conversations(captions_path: str, output_path: str):
    with open(captions_path, "r") as f:
        captions = [json.loads(line) for line in f]

    with open("data_gen/prompts/caption_instructions.jsonl", "r") as f:
        instructions = [json.loads(line) for line in f]
        instruction_pool = [instruction["instruction"] for instruction in instructions]

    outputs = []
    id = 0
    for data in tqdm(captions, desc="Formatting captions data"):
        if not data["caption"]:
            continue
        if "\n\n\n" in data["caption"]:  # Ignore thinking part
            caption = data["caption"].split("\n\n\n")[-1]
        else:
            caption = data["caption"]

        instruction = random.choice(instruction_pool)
        image_token_pos = random.choice(["start", "end"])
        if image_token_pos == "start":
            instruction = f"<image>\n{instruction}"
        elif image_token_pos == "end":
            instruction = f"{instruction}\n<image>"

        conversations = [{"from": "human", "value": f"{instruction}"}, {"from": "gpt", "value": caption}]

        output = {
            "id": f"{id:09d}",
            "image": data["image"],
            "conversations": conversations,
        }
        outputs.append(output)
        id += 1

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)


def format_mmcs_to_conversations(
    mmcs_results_path: str, output_path: str, add_whole_image: bool = True, captions_only: bool = False
):
    with open(mmcs_results_path, "r") as f:
        mmcs_results = [json.loads(line) for line in f]

    with open("data_gen/prompts/caption_instructions.jsonl", "r") as f:
        instructions = [json.loads(line) for line in f]
        instruction_pool = [instruction["instruction"] for instruction in instructions]

    outputs = []
    id = 0
    for data in tqdm(mmcs_results, desc="Formatting MMCS data"):
        conversations = []
        if add_whole_image:
            instruction = random.choice(instruction_pool)
            image_token_pos = random.choice(["start", "end"])
            if image_token_pos == "start":
                instruction = f"<image>\n{instruction}"
            elif image_token_pos == "end":
                instruction = f"{instruction}\n<image>"
            conversations.append({"from": "human", "value": f"{instruction}"})

        conversations.append({"from": "gpt", "value": data["caption"]})

        if captions_only:
            output = {
                "id": f"{id:09d}",
                "image": data["image"],
                "conversations": conversations,
            }
        else:
            output = {
                "id": f"{id:09d}",
                "image": data["image"],
                "conversations": conversations,
                "annotations": data["annotations"],
            }
        outputs.append(output)
        id += 1

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    data_dir = "data/annotations"
    dataset_name = "sharecaptioner_600k"

    # format_mmcs_to_conversations(
    #     mmcs_results_path=os.path.join(data_dir, f"{dataset_name}.jsonl"),
    #     output_path=os.path.join(data_dir, f"{dataset_name}_llava.json"),
    #     add_whole_image=False,
    #     captions_only=False,
    # )

    format_mmcs_to_conversations(
        mmcs_results_path=os.path.join(data_dir, f"{dataset_name}.jsonl"),
        output_path=os.path.join("data/captions", f"{dataset_name}_llava.json"),
        add_whole_image=True,
        captions_only=True,
    )

    # format_mmcs_to_conversations(
    #     mmcs_results_path=os.path.join(data_dir, f"{dataset_name}.jsonl"),
    #     output_path=os.path.join("data/patch_align", f"{dataset_name}_llava.json"),
    #     add_whole_image=True,
    #     captions_only=False,
    # )
