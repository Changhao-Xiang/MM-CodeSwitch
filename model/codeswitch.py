from typing import List

import numpy as np
import torch


def extract_mask_patch_features(image_features: torch.Tensor, mask_list: List[np.ndarray]):
    """
    Extract the features for the patches inside the mask.
    """

    assert image_features.ndim == 2
    num_patches, hidden_size = image_features.shape
    patches_per_dim = int((num_patches) ** 0.5)  # Assuming square image patches layout

    extracted_features = []
    # Find all patches that are covered by the mask
    for mask in mask_list:
        height, width = mask.shape
        # Find all patches that are covered by the mask
        # Vectorized conversion of pixel coordinates to patch indices
        mask_y_coords, mask_x_coords = np.where(mask == 1)
        patch_y = np.minimum((mask_y_coords / height * patches_per_dim).astype(int), patches_per_dim - 1)
        patch_x = np.minimum((mask_x_coords / width * patches_per_dim).astype(int), patches_per_dim - 1)
        patch_indices = set(patch_y * patches_per_dim + patch_x)

        extracted_features.append(image_features[sorted(list(patch_indices)), :])

    if len(extracted_features) > 0:
        return torch.cat(extracted_features, dim=0)
    else:
        print(f"No mask patches found. mask_list: {[mask.sum() for mask in mask_list]}")
        return torch.zeros((0, image_features.shape[1]), device=image_features.device, dtype=image_features.dtype)


def extract_bbox_patch_features(image_features: torch.Tensor, bbox_list: List[List[float]]):
    """
    Extract the features for the patches inside the bbox.

    Args:
        image_features: Tensor of shape [num_patches, hidden_size] containing image patch features
        bbox: List of floats containing the bounding box of the patches, in the format [x, y, w, h] with relative coordinates

    Returns:
        Tensor of shape [num_patches, hidden_size] containing the features of the patches inside the bbox
    """
    assert image_features.ndim == 2
    num_patches, hidden_size = image_features.shape

    extracted_features = []
    for bbox in bbox_list:
        # Calculate the number of patches in each dimension
        x1, y1, x2, y2 = bbox
        patches_per_dim = int((num_patches) ** 0.5)  # Assuming square image patches layout

        # Convert relative coordinates to patch indices
        x_start = int(x1 * patches_per_dim)
        y_start = int(y1 * patches_per_dim)
        x_end = min(int(x2 * patches_per_dim) + 1, patches_per_dim)
        y_end = min(int(y2 * patches_per_dim) + 1, patches_per_dim)

        # Create a mask for the patches inside the bbox
        patch_indices = []
        for i in range(y_start, y_end):
            for j in range(x_start, x_end):
                # Convert 2D coordinates to 1D index
                patch_idx = i * patches_per_dim + j
                if patch_idx < num_patches:  # Ensure index is valid
                    patch_indices.append(patch_idx)

        # Extract the features for the selected patches
        extracted_features.append(image_features[patch_indices, :])

    return torch.cat(extracted_features, dim=0)


def extract_patch_features_by_indices(image_features: torch.Tensor, patch_indices: List[int]):
    """
    Extract the features for the patches by their indices.

    Args:
        image_features: Tensor of shape [num_patches, hidden_size] containing image patch features
        patch_indices: List of integers containing the patch indices to extract

    Returns:
        Tensor of shape [len(patch_indices), hidden_size] containing the features of the specified patches
    """
    assert image_features.ndim == 2
    if not patch_indices:
        raise ValueError("Empty object patch indices!")
    if any(idx < 0 or idx >= image_features.shape[0] for idx in patch_indices):
        raise ValueError(
            f"Invalid object patch indices: {patch_indices} for image features of shape {image_features.shape}"
        )

    return image_features[patch_indices, :]


def find_ids_positions(input_ids: torch.Tensor, target_ids: torch.Tensor):
    """
    Find the positions of the longest common substring between input_ids and target_ids.

    Args:
        input_ids: 1D tensor containing the full sequence of token IDs
        target_ids: 1D tensor containing the sequence to compare with

    Returns:
        1D tensor containing the positions in input_ids that form the longest common substring
    """
    assert input_ids.ndim == 1 and target_ids.ndim == 1

    # Get lengths of both sequences
    input_len = input_ids.size(0)
    target_len = target_ids.size(0)

    # If either sequence is empty, return empty tensor
    if input_len == 0 or target_len == 0:
        return torch.tensor([], dtype=torch.long, device=input_ids.device)

    # Convert tensors to CPU and numpy for easier processing
    input_ids_np = input_ids.cpu().numpy()
    target_ids_np = target_ids.cpu().numpy()

    # Create a dynamic programming table to find the longest common substring
    dp = torch.zeros((input_len + 1, target_len + 1), dtype=torch.long)

    # Variables to track the longest substring
    max_length = 0
    end_pos = 0

    # Fill the dp table
    for i in range(1, input_len + 1):
        for j in range(1, target_len + 1):
            if input_ids_np[i - 1] == target_ids_np[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
                if dp[i, j] > max_length:
                    max_length = dp[i, j]
                    end_pos = i

    # If no common substring found, return empty tensor
    if max_length == 0:
        return None

    # Calculate the start position of the longest common substring
    start_pos = end_pos - max_length

    # Create a tensor with the positions of the longest common substring
    positions = np.arange(start_pos, end_pos)

    return torch.tensor(positions, dtype=torch.long, device=input_ids.device)


def rearange_patch_features(patch_features: torch.Tensor, positions: torch.Tensor):
    """
    Rearrange the patch features to match the positions of positions.

    This function distributes patch features across the positions by grouping
    patches and averaging their features when necessary.

    Args:
        patch_features: Tensor of shape [num_patches, hidden_size] containing image patch features
        positions: Tensor containing the positions where features should be placed

    Returns:
        Tensor of shape [num_positions, hidden_size] with rearranged features
    """
    num_patches, hidden_size = patch_features.shape
    num_positions = positions.size(0)

    # If there are no positions, return empty tensor
    if num_positions == 0:
        return torch.zeros((0, hidden_size), device=patch_features.device, dtype=patch_features.dtype)

    # If there are no patches, return zeros for all positions
    if num_patches == 0:
        return torch.zeros((num_positions, hidden_size), device=positions.device, dtype=patch_features.dtype)

    # Calculate how many patches should be assigned to each position
    patches_per_position = num_patches / num_positions

    # Initialize tensor to store rearranged features
    rearranged_features = torch.zeros(
        (num_positions, hidden_size), device=patch_features.device, dtype=patch_features.dtype
    )

    # Distribute patches across positions
    for i in range(num_positions):
        # Calculate start and end indices for this group of patches
        start_idx = int(i * patches_per_position)
        end_idx = int((i + 1) * patches_per_position)

        # Handle edge case for the last group
        if i == num_positions - 1:
            end_idx = num_patches

        # If there are patches for this position
        if start_idx < end_idx:
            # Average the features of all patches assigned to this position
            rearranged_features[i] = torch.mean(patch_features[start_idx:end_idx], dim=0)

    return rearranged_features


if __name__ == "__main__":
    pass
