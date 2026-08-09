import json
import os
from typing import List

from tqdm import tqdm


def merge_json_files(file_paths: list[str], output_path: str):
    """
    Combine multiple JSON files into a single output file.
    """
    combined_data = []

    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    try:
        with open(output_path, "w") as f:
            json.dump(combined_data, f, indent=4)
        print(f"Successfully combined {len(file_paths)} files into {output_path}")
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")


def merge_jsonl_files(file_paths: list[str], output_path: str) -> None:
    """
    Merge multiple jsonl files into one, with deduplication based on 'image' field.

    Args:
        input_files: List of paths to input jsonl files
        output_file: Path to the output jsonl file
    """
    total_num_samples = 0
    skipped_duplicates = 0
    seen_images = set()

    with open(output_path, "w", encoding="utf-8") as outfile:
        for file_path in tqdm(file_paths):
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        if line.strip():  # Skip empty lines
                            try:
                                data = json.loads(line)
                                image_field = data.get("image")

                                # Skip if image field already seen
                                if image_field and image_field in seen_images:
                                    skipped_duplicates += 1
                                    continue

                                # Add image to seen set and write the line
                                if image_field:
                                    seen_images.add(image_field)

                                outfile.write(line)
                                total_num_samples += 1
                            except json.JSONDecodeError:
                                print(f"Warning: Invalid JSON line in {file_path}. Skipping line.")
            except FileNotFoundError:
                print(f"Warning: File {file_path} not found. Skipping.")

    print(f"Total number of samples: {total_num_samples}")
    print(f"Skipped duplicates: {skipped_duplicates}")


if __name__ == "__main__":
    # Get all jsonl files in the data directory
    data_dir = "data/segment_new"
    dataset_name = "gqa_120k"
    file_list = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(dataset_name) and f.endswith(".jsonl")
    ]

    merge_jsonl_files(file_list, os.path.join(data_dir, f"{dataset_name}.jsonl"))
