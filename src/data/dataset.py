"""
NEU Surface Defect Dataset - PyTorch Dataset class.

Loads the NEU Surface Defect Dataset (1,800 grayscale images, 200x200px)
organized into 6 class subfolders. Supports both classification and
detection tasks.

Usage:
    from src.data.dataset import NEUDataset

    dataset = NEUDataset(root_dir="data/raw/images", transform=my_transforms)
    image, label = dataset[0]
    print(dataset.get_stats())
"""

import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import CLASS_NAMES, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NEUDataset(Dataset):
    """
    PyTorch Dataset for the NEU Surface Defect Dataset.

    Loads grayscale images from class-organized subdirectories and
    converts them to 3-channel (RGB) format for pretrained model compatibility.

    Args:
        root_dir: Path to the root images directory containing class subfolders.
        transform: Optional albumentations or torchvision transform pipeline.
        class_names: List of class names (default: from config).
        grayscale: If True, keep images as single-channel grayscale.

    Attributes:
        samples: List of (image_path, label_index) tuples.
        class_to_idx: Mapping from class name to integer label.
        idx_to_class: Mapping from integer label to class name.
    """

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        class_names: Optional[list[str]] = None,
        grayscale: bool = False,
    ):
        self.root_dir = Path(root_dir) if root_dir else RAW_DATA_DIR / "images"
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.grayscale = grayscale

        # Build class-to-index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        # Load all image paths and labels
        self.samples = self._load_samples()

        logger.info(
            f"NEUDataset initialized: {len(self.samples)} images, "
            f"{len(self.class_names)} classes, root={self.root_dir}"
        )

    def _load_samples(self) -> list[tuple[Path, int]]:
        """
        Scan class subdirectories and build list of (path, label) tuples.

        Returns:
            List of (image_path, class_index) tuples sorted by class then filename.
        """
        samples = []
        image_extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            label = self.class_to_idx[class_name]
            class_images = sorted([
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ])

            for img_path in class_images:
                samples.append((img_path, label))

        if not samples:
            logger.error(
                f"No images found in {self.root_dir}. "
                "Ensure the dataset is downloaded and organized."
            )

        return samples

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load and return a single sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (image_tensor, label_index).
            Image tensor shape: (3, H, W) for RGB or (1, H, W) for grayscale.
        """
        img_path, label = self.samples[idx]

        # Load image with OpenCV (reads as BGR)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to load image: {img_path}")

        # Convert grayscale to 3-channel for pretrained models
        if not self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply transforms
        if self.transform is not None:
            # Support albumentations-style transforms
            if hasattr(self.transform, "transform") or hasattr(self.transform, "__call__"):
                try:
                    # Albumentations interface
                    augmented = self.transform(image=image)
                    image = augmented["image"]
                except (TypeError, KeyError):
                    # torchvision-style transform
                    from PIL import Image as PILImage
                    pil_image = PILImage.fromarray(image)
                    image = self.transform(pil_image)
                    return image, label

        # Convert numpy array to tensor if not already
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale: (H, W) -> (1, H, W)
                image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            else:
                # RGB: (H, W, 3) -> (3, H, W)
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    def get_image_path(self, idx: int) -> Path:
        """Get the file path for a specific sample index."""
        return self.samples[idx][0]

    def get_class_name(self, idx: int) -> str:
        """Get the class name for a specific sample index."""
        _, label = self.samples[idx]
        return self.idx_to_class[label]

    def get_stats(self) -> dict:
        """
        Compute and return dataset statistics.

        Returns:
            Dictionary with keys: total_images, num_classes, class_distribution,
            image_extensions, sample_resolution.
        """
        class_counts = {}
        extensions = set()

        for img_path, label in self.samples:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            extensions.add(img_path.suffix.lower())

        # Get sample resolution from first image
        sample_resolution = None
        if self.samples:
            img = cv2.imread(str(self.samples[0][0]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sample_resolution = img.shape  # (H, W)

        return {
            "total_images": len(self.samples),
            "num_classes": len(self.class_names),
            "class_distribution": class_counts,
            "image_extensions": list(extensions),
            "sample_resolution": sample_resolution,
            "root_directory": str(self.root_dir),
        }

    def get_class_samples(self, class_name: str, n: int = 5) -> list[tuple[np.ndarray, int]]:
        """
        Get N sample images from a specific class.

        Args:
            class_name: Name of the defect class.
            n: Number of samples to return.

        Returns:
            List of (image_array, label) tuples.
        """
        if class_name not in self.class_to_idx:
            raise ValueError(
                f"Unknown class: {class_name}. Available: {self.class_names}"
            )

        label = self.class_to_idx[class_name]
        class_samples = [(p, l) for p, l in self.samples if l == label]

        results = []
        for img_path, lbl in class_samples[:n]:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                results.append((image, lbl))

        return results

    def get_all_labels(self) -> list[int]:
        """Return all labels as a list (useful for stratified splitting)."""
        return [label for _, label in self.samples]

    def get_all_paths(self) -> list[Path]:
        """Return all image paths."""
        return [path for path, _ in self.samples]
