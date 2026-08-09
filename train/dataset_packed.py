import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from common.args import training_args
from model.codeswitch import find_ids_positions
from model.constants import IGNORE_INDEX
from model.dynamic_resolution import dynamic_preprocess


class LlavaDataset(Dataset):
    def __init__(self, data_recipe_path, processor, vision_config):
        self.sources = []
        for recipe in json.load(open(data_recipe_path, "r")):
            annotation = json.load(open(recipe["annotation"], "r"))

            # Sample annotation for each dataset
            sample_ratio = recipe["sample_ratio"]
            num_samples = int(len(annotation) * sample_ratio)
            annotation = random.sample(annotation, num_samples)

            for anno in annotation:
                assert isinstance(anno, list), f"Packed data should be a list of samples, but got {type(anno)}"
                for i, sample in enumerate(anno):
                    if "image" in sample:
                        anno[i]["image"] = os.path.join(recipe["image_folder"], sample["image"])
                self.sources.append(anno)

        self.processor = processor

        # Config dynamic resolution
        self.dynamic_resolution = vision_config.dynamic_resolution
        if self.dynamic_resolution == "native":
            self.processor.image_processor.max_pixels = training_args.max_pixels
            self.processor.image_processor.min_pixels = training_args.min_pixels
            self.processor.image_processor.size["longest_edge"] = training_args.max_pixels
            self.processor.image_processor.size["shortest_edge"] = training_args.min_pixels
            self.merge_size = self.processor.image_processor.merge_size
        elif self.dynamic_resolution == "tile":
            self.tile_size = vision_config.image_size
            self.patch_size = vision_config.patch_size
            self.max_num_tiles = vision_config.max_num_tiles
            self.merge_size = 2
        else:
            raise ValueError(f"Invalid dynamic resolution: {self.dynamic_resolution}")

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index):
        sources = self.sources[index]

        # Expect packed data format: List[Dict]
        if isinstance(sources, dict):
            # Convert single sample to list for uniform processing
            sources = [sources]
        elif not isinstance(sources, list):
            raise ValueError(f"Expected list of samples for packed data, got {type(sources)}")

        # Process packed samples
        return self._process_packed_samples(sources)

    def _process_packed_samples(self, sources):
        """Process multiple samples for packed data"""
        sample_data_list = []

        # Process each sample individually
        for source in sources:
            sample_data = self._process_single_sample_inline(source)
            sample_data_list.append(sample_data)

        # Concatenate all samples
        return self._concatenate_samples(sample_data_list)

    def _process_single_sample_inline(self, source):
        """Process a single sample inline for packed data"""
        data_dict = {}

        # Process image into pixel_values
        if "image" in source:
            image = Image.open(source["image"]).convert("RGB")
            if self.dynamic_resolution == "native":
                visual_processed = self.processor.image_processor.preprocess(image, return_tensors="pt")
                pixel_values = visual_processed["pixel_values"]
                image_grid_thw = visual_processed["image_grid_thw"]
                data_dict["image_grid_thw"] = image_grid_thw
                # 28 * 28 per patch(tile), num_tiles == num_patches
                num_image_tiles = pixel_values.shape[0] // (self.merge_size**2)
                target_aspect_ratio = (
                    int(image_grid_thw[0][2] // self.merge_size),
                    int(image_grid_thw[0][1] // self.merge_size),
                )
            elif self.dynamic_resolution == "tile":
                processed_images, target_aspect_ratio = dynamic_preprocess(
                    image, tile_size=self.tile_size, max_num_tiles=self.max_num_tiles
                )
                pixel_values = self.processor.image_processor(processed_images, return_tensors="pt")["pixel_values"]
                num_image_tiles = len(processed_images)
            data_dict["pixel_values"] = pixel_values
        else:
            num_image_tiles = 0
            data_dict["pixel_values"] = None
            target_aspect_ratio = (1, 1)  # Default for no image

        # Process conversations
        input_ids, label_ids = self.processor.preprocess(source["conversations"], num_image_tiles)
        data_dict["input_ids"] = input_ids
        data_dict["label_ids"] = label_ids
        data_dict["num_image_tiles_batch"] = num_image_tiles

        # Store sequence length for attention mask processing
        data_dict["attention_mask"] = [data_dict["input_ids"].size(0)]

        return data_dict

    def _concatenate_samples(self, sample_data_list):
        """Concatenate multiple processed samples into packed format"""
        # Concatenate input_ids and label_ids
        input_ids = torch.cat([data["input_ids"] for data in sample_data_list], dim=0)
        label_ids = torch.cat([data["label_ids"] for data in sample_data_list], dim=0)

        # Collect sequence lengths for attention mask
        seq_lengths = [data["attention_mask"][0] for data in sample_data_list]

        # Handle pixel_values - concatenate all images
        pixel_values_list = [data["pixel_values"] for data in sample_data_list if data["pixel_values"] is not None]
        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None

        # Handle image_grid_thw
        image_grid_thw_list = [data["image_grid_thw"] for data in sample_data_list if "image_grid_thw" in data]
        image_grid_thw = torch.cat(image_grid_thw_list, dim=0) if image_grid_thw_list else None

        # Collect num_image_tiles_batch and patch_indices_list_batch
        num_image_tiles_batch = [data["num_image_tiles_batch"] for data in sample_data_list]
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": seq_lengths,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_image_tiles_batch": num_image_tiles_batch,
        }


@dataclass
class PackedDataCollator:
    """Optimized data collator specifically for packed data"""

    pad_token_id: int
    model_max_length: int
    merge_size: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate packed data features into batch"""
        # Collect all sequences (already concatenated within each packed sample)
        input_ids = [feature["input_ids"] for feature in features]
        label_ids = [feature["label_ids"] for feature in features]

        # Collect sequence lengths for packed sequences
        all_seq_lengths = []
        num_image_tiles_batch = []
        for feature in features:
            all_seq_lengths.extend(feature["attention_mask"])
            num_image_tiles_batch.extend(feature["num_image_tiles_batch"])

        # Create cumulative sequence lengths for attention mask
        seq_lens = torch.tensor([0] + all_seq_lengths, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        # Concatenate all sequences across batch
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)  # [1, total_length]
        label_ids = torch.cat(label_ids, dim=0).unsqueeze(0)  # [1, total_length]

        # Generate position_ids for packed sequences
        position_ids = self._generate_packed_position_ids(all_seq_lengths)

        # Truncate if necessary
        if input_ids.size(1) > self.model_max_length:
            input_ids = input_ids[:, : self.model_max_length]
            label_ids = label_ids[:, : self.model_max_length]
            position_ids = position_ids[:, : self.model_max_length]
            # Adjust cumsum_seq_lens accordingly
            cumsum_seq_lens = cumsum_seq_lens[cumsum_seq_lens <= self.model_max_length]
            if len(cumsum_seq_lens) == 0 or cumsum_seq_lens[-1] < self.model_max_length:
                cumsum_seq_lens = torch.cat([cumsum_seq_lens, torch.tensor([self.model_max_length], dtype=torch.int32)])

        # Handle pixel_values - concatenate all images
        pixel_values_list = [feature["pixel_values"] for feature in features if feature["pixel_values"] is not None]
        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None

        # Handle image_grid_thw
        if "image_grid_thw" in features[0]:
            image_grid_thw_list = [
                feature["image_grid_thw"] for feature in features if feature["image_grid_thw"] is not None
            ]
            image_grid_thw = torch.cat(image_grid_thw_list, dim=0) if image_grid_thw_list else None
        else:
            image_grid_thw = None

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask": cumsum_seq_lens,
            "position_ids": position_ids,
            "num_image_tiles_batch": [sum(num_image_tiles_batch)],
        }

    def _generate_packed_position_ids(self, seq_lengths):
        """Generate position_ids for packed sequences where each sequence starts from 0"""
        position_ids = []
        for length in seq_lengths:
            position_ids.extend(list(range(length)))
        return torch.tensor(position_ids, dtype=torch.long).unsqueeze(0)  # [1, total_length]


def make_packed_data_module(data_recipe_path, processor, vision_config):
    """Create dataset and data collator for packed data training"""
    train_dataset = LlavaDataset(data_recipe_path=data_recipe_path, processor=processor, vision_config=vision_config)

    data_collator = PackedDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        model_max_length=training_args.model_max_length,
        merge_size=train_dataset.merge_size,
    )

    return train_dataset, data_collator
