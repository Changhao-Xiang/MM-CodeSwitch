import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pycocotools.mask as mask_util
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
                if "image" in anno:
                    anno["image"] = os.path.join(recipe["image_folder"], anno["image"])
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

        self.mmcs_type = training_args.mmcs_type  # bbox or mask

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index):
        source = self.sources[index]

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
        # Number of image tiles for this sample
        data_dict["num_image_tiles_batch"] = num_image_tiles

        # Captions only
        if not training_args.enable_mmcs or "annotations" not in source:
            return data_dict

        # Process input_ids and label_ids for mmcs annotations
        new_input_ids, new_label_ids, patch_indices_list = self.process_annotations(
            source, input_ids, label_ids, image.size, target_aspect_ratio
        )
        data_dict["input_ids"] = new_input_ids
        data_dict["label_ids"] = new_label_ids
        # A list of patch indices for objects in the image
        data_dict["patch_indices_list_batch"] = patch_indices_list
        return data_dict

    def process_annotations(self, source, input_ids, label_ids, image_size, target_aspect_ratio):
        cur_input_ids = input_ids.clone()
        cur_label_ids = label_ids.clone()
        patch_indices_list = []

        for annotation in source["annotations"]:
            if self.dynamic_resolution == "native":
                num_image_patches, patch_indices = self.get_native_res_patch_indices(
                    annotation, image_size, target_aspect_ratio
                )
            elif self.dynamic_resolution == "tile":
                num_image_patches, patch_indices = self.get_tile_patch_indices(
                    annotation, image_size, target_aspect_ratio
                )
            else:
                raise ValueError(f"Invalid dynamic resolution: {self.dynamic_resolution}")

            if num_image_patches == 0:
                print(f"No patches found for {source['image']}")
                continue

            obj_str = annotation["object"]
            obj_ids = self.processor.tokenizer.encode(obj_str, add_special_tokens=False, return_tensors="pt").view(-1)
            obj_positions = find_ids_positions(cur_input_ids, obj_ids)
            if obj_positions is None or len(obj_positions) <= 1:
                # Retry with addtional space before obj_str
                obj_ids = self.processor.tokenizer.encode(
                    " " + obj_str, add_special_tokens=False, return_tensors="pt"
                ).view(-1)
                obj_positions = find_ids_positions(cur_input_ids, obj_ids)

            # If no common substring found, skip codeswitch for current obj
            if obj_positions is None:
                continue

            # Record the patch indices for this object
            patch_indices_list.append(patch_indices)

            # Insert [image_token_id] * num_mask_patches at obj_positions
            new_input_ids = cur_input_ids[: obj_positions[0]]
            new_label_ids = cur_label_ids[: obj_positions[0]]
            new_input_ids = torch.cat(
                [
                    new_input_ids,
                    torch.full(
                        (num_image_patches,),
                        self.processor.image_token_id,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                ]
            )
            if num_image_patches < len(obj_positions):
                obj_positions = obj_positions[:num_image_patches]
            new_label_ids = torch.cat(
                [
                    new_label_ids,
                    torch.full(
                        (num_image_patches - len(obj_positions),),
                        IGNORE_INDEX,
                        dtype=torch.long,
                        device=label_ids.device,
                    ),
                ]
            )
            new_input_ids = torch.cat([new_input_ids, cur_input_ids[obj_positions[-1] + 1 :]])
            new_label_ids = torch.cat([new_label_ids, cur_label_ids[obj_positions[0] :]])
            assert new_input_ids.shape == new_label_ids.shape

            cur_input_ids = new_input_ids
            cur_label_ids = new_label_ids

        return cur_input_ids, cur_label_ids, patch_indices_list

    def get_tile_patch_indices(self, annotation, image_size, target_aspect_ratio):
        if self.mmcs_type == "bbox":
            return self._get_tile_patch_indices_bbox(annotation, image_size, target_aspect_ratio)
        elif self.mmcs_type == "mask":
            return self._get_tile_patch_indices_mask(annotation, image_size, target_aspect_ratio)

    def _get_tile_patch_indices_bbox(self, annotation, image_size, target_aspect_ratio):
        patch_indices = []
        # Assuming square image patches layout per tile
        patches_per_dim = self.tile_size // (self.patch_size * self.merge_size)
        num_patches_per_tile = patches_per_dim**2

        bbox = annotation["bbox"]
        x1, y1, x2, y2 = bbox
        # Convert to relative coordinates
        x1_rel, y1_rel = x1 / image_size[0], y1 / image_size[1]
        x2_rel, y2_rel = x2 / image_size[0], y2 / image_size[1]

        # Get tile dimensions
        num_tiles_w, num_tiles_h = target_aspect_ratio
        # For each tile, check if the bbox intersects with it
        for tile_h in range(num_tiles_h):
            for tile_w in range(num_tiles_w):
                # Calculate tile boundaries in relative coordinates
                tile_x1, tile_y1 = tile_w / num_tiles_w, tile_h / num_tiles_h
                tile_x2, tile_y2 = (tile_w + 1) / num_tiles_w, (tile_h + 1) / num_tiles_h

                # Check if bbox intersects with this tile
                if not (x2_rel <= tile_x1 or x1_rel >= tile_x2 or y2_rel <= tile_y1 or y1_rel >= tile_y2):
                    # Calculate intersection
                    intersect_x1, intersect_y1 = max(x1_rel, tile_x1), max(y1_rel, tile_y1)
                    intersect_x2, intersect_y2 = min(x2_rel, tile_x2), min(y2_rel, tile_y2)

                    # Convert intersection to tile-local coordinates (0-1)
                    local_x1 = (intersect_x1 - tile_x1) / (tile_x2 - tile_x1)
                    local_y1 = (intersect_y1 - tile_y1) / (tile_y2 - tile_y1)
                    local_x2 = (intersect_x2 - tile_x1) / (tile_x2 - tile_x1)
                    local_y2 = (intersect_y2 - tile_y1) / (tile_y2 - tile_y1)

                    # Convert to patch indices within this tile
                    patch_x_start = int(local_x1 * patches_per_dim)
                    patch_y_start = int(local_y1 * patches_per_dim)
                    patch_x_end = min(int(local_x2 * patches_per_dim) + 1, patches_per_dim)
                    patch_y_end = min(int(local_y2 * patches_per_dim) + 1, patches_per_dim)

                    # Calculate global patch indices
                    tile_idx = tile_h * num_tiles_w + tile_w
                    tile_offset = tile_idx * num_patches_per_tile
                    for i in range(patch_y_start, patch_y_end):
                        for j in range(patch_x_start, patch_x_end):
                            patch_idx = tile_offset + i * patches_per_dim + j
                            patch_indices.append(patch_idx)

        return len(patch_indices), patch_indices

    def _get_tile_patch_indices_mask(self, annotation, image_size, target_aspect_ratio):
        patch_indices = []
        # Assuming square image patches layout per tile
        patches_per_dim = self.tile_size // (self.patch_size * self.merge_size)
        num_patches_per_tile = patches_per_dim**2

        mask = mask_util.decode(annotation["segmentation"])
        height, width = mask.shape

        # Get tile dimensions
        num_tiles_w, num_tiles_h = target_aspect_ratio
        # For each tile, check which patches are covered by the mask
        for tile_h in range(num_tiles_h):
            for tile_w in range(num_tiles_w):
                # Calculate tile boundaries in pixel coordinates
                tile_x1 = int(tile_w * width / num_tiles_w)
                tile_y1 = int(tile_h * height / num_tiles_h)
                tile_x2 = int((tile_w + 1) * width / num_tiles_w)
                tile_y2 = int((tile_h + 1) * height / num_tiles_h)

                # Extract mask for this tile
                tile_mask = mask[tile_y1:tile_y2, tile_x1:tile_x2]
                if tile_mask.sum() > 0:  # If there's any mask in this tile
                    tile_mask_height, tile_mask_width = tile_mask.shape
                    # Find patches covered by the mask in this tile
                    mask_y_coords, mask_x_coords = np.where(tile_mask == 1)
                    if len(mask_y_coords) > 0:
                        patch_y = np.minimum(
                            (mask_y_coords / tile_mask_height * patches_per_dim).astype(int),
                            patches_per_dim - 1,
                        )
                        patch_x = np.minimum(
                            (mask_x_coords / tile_mask_width * patches_per_dim).astype(int),
                            patches_per_dim - 1,
                        )

                        # Calculate global patch indices
                        tile_idx = tile_h * num_tiles_w + tile_w
                        tile_offset = tile_idx * num_patches_per_tile
                        tile_patch_indices_set = set(tile_offset + patch_y * patches_per_dim + patch_x)
                        patch_indices.extend(list(tile_patch_indices_set))

        # Remove duplicates and sort
        patch_indices = sorted(list(set(patch_indices)))
        return len(patch_indices), patch_indices

    def get_native_res_patch_indices(self, annotation, image_size, target_aspect_ratio):
        assert self.mmcs_type == "bbox", "mmcs_mask not implemented yet"
        patch_indices = []
        patches_w, patches_h = target_aspect_ratio

        bbox = annotation["bbox"]
        x1, y1, x2, y2 = bbox
        # Convert to relative coordinates
        x1_rel = x1 / image_size[0]
        y1_rel = y1 / image_size[1]
        x2_rel = x2 / image_size[0]
        y2_rel = y2 / image_size[1]

        # Convert relative coordinates to patch indices
        x_start = int(x1_rel * patches_w)
        y_start = int(y1_rel * patches_h)
        x_end = min(int(x2_rel * patches_w) + 1, patches_w)
        y_end = min(int(y2_rel * patches_h) + 1, patches_h)

        for i in range(y_start, y_end):
            for j in range(x_start, x_end):
                # Convert 2D coordinates to 1D index
                patch_idx = i * patches_w + j
                patch_indices.append(patch_idx)

        return len(patch_indices), patch_indices


@dataclass
class DataCollatorWithPadding:
    pad_token_id: int
    model_max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [feature["input_ids"] for feature in features]
        label_ids = [feature["label_ids"] for feature in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.model_max_length]
        label_ids = label_ids[:, : self.model_max_length]

        # Filter out None values and stack the pixel_values into a tensor
        pixel_values = [feature["pixel_values"] for feature in features if feature["pixel_values"] is not None]
        if pixel_values:
            pixel_values = torch.concat(pixel_values, dim=0)
        else:
            pixel_values = None

        image_grid_thw = None
        if any("image_grid_thw" in feature for feature in features):
            image_grid_thw = [feature["image_grid_thw"] for feature in features if "image_grid_thw" in feature]
            image_grid_thw = torch.cat(image_grid_thw, dim=0)

        patch_indices_list_batch = []
        num_image_tiles_batch = []
        for feature in features:
            if "patch_indices_list_batch" in feature:
                patch_indices_list_batch.append(feature["patch_indices_list_batch"])
            else:
                patch_indices_list_batch.append(None)

            if "num_image_tiles_batch" in feature:
                num_image_tiles_batch.append(feature["num_image_tiles_batch"])
            else:
                num_image_tiles_batch.append(0)

        batch = dict(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            labels=label_ids,
            attention_mask=input_ids.ne(self.pad_token_id),
            patch_indices_list_batch=patch_indices_list_batch,
            num_image_tiles_batch=num_image_tiles_batch,
        )

        return batch


def make_data_module(data_recipe_path, processor, vision_config):
    """Create dataset and data collator"""
    train_dataset = LlavaDataset(
        data_recipe_path=data_recipe_path,
        processor=processor,
        vision_config=vision_config,
    )

    data_collator = DataCollatorWithPadding(
        pad_token_id=processor.tokenizer.pad_token_id, model_max_length=training_args.model_max_length
    )

    return train_dataset, data_collator
