"""
Data preprocessing utilities for the NEU Surface Defect Dataset.

Handles image resizing, normalization, channel conversion, and
dataset splitting with stratification.

Usage:
    from src.data.preprocessing import preprocess_image, get_transforms
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from config.settings import (
    IMAGE_SIZE_CLASSIFICATION,
    IMAGE_SIZE_DETECTION,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = IMAGE_SIZE_CLASSIFICATION,
    normalize: bool = True,
    to_tensor: bool = True,
) -> np.ndarray | torch.Tensor:
    """
    Preprocess a single image for model inference.

    Steps:
        1. Convert grayscale to 3-channel RGB (if needed)
        2. Resize to target dimensions
        3. Normalize with ImageNet statistics
        4. Convert to PyTorch tensor (C, H, W)

    Args:
        image: Input image (grayscale or RGB numpy array).
        target_size: Target (height, width) dimensions.
        normalize: Whether to apply ImageNet normalization.
        to_tensor: Whether to convert to PyTorch tensor.

    Returns:
        Preprocessed image as tensor (C, H, W) or numpy array (H, W, C).
    """
    # Ensure 3-channel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize
    image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # Convert to float [0, 1]
    image = image.astype(np.float32) / 255.0

    # Normalize
    if normalize:
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        image = (image - mean) / std

    # Convert to tensor
    if to_tensor:
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    return image


def get_classification_transforms(train: bool = True) -> transforms.Compose:
    """
    Get torchvision transforms for classification models.

    Args:
        train: If True, includes augmentation; otherwise only resize + normalize.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE_CLASSIFICATION),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE_CLASSIFICATION),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def denormalize_image(
    tensor: torch.Tensor,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W).
        mean: Normalization mean values.
        std: Normalization std values.

    Returns:
        Denormalized numpy image (H, W, C) in [0, 255] uint8.
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor.clone().detach().cpu()
    else:
        img = torch.tensor(tensor)

    # Denormalize
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    # Clamp and convert
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return img


def split_dataset_indices(
    labels: list[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset indices with stratification.

    Ensures each split maintains the same class distribution as the original.

    Args:
        labels: List of integer labels for all samples.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    indices = list(range(len(labels)))

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=[labels[i] for i in indices],
        random_state=seed,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=relative_test_ratio,
        stratify=[labels[i] for i in temp_indices],
        random_state=seed,
    )

    logger.info(
        f"Dataset split: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    return train_indices, val_indices, test_indices
