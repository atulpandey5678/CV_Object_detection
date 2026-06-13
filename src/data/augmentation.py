"""
Data augmentation pipelines using albumentations.

Provides configurable augmentation strategies for training, validation, and
test sets. Aggressive augmentation is critical for the small NEU dataset
(only 300 images per class).

Usage:
    from src.data.augmentation import get_train_augmentation, get_val_augmentation

    train_transform = get_train_augmentation()
    val_transform = get_val_augmentation()
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.settings import IMAGE_SIZE_CLASSIFICATION, IMAGE_SIZE_DETECTION, IMAGENET_MEAN, IMAGENET_STD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_train_augmentation(
    img_size: tuple[int, int] = IMAGE_SIZE_CLASSIFICATION,
    use_advanced: bool = True,
) -> A.Compose:
    """
    Get training augmentation pipeline with albumentations.

    Augmentations applied:
        - Resize to target size
        - Random rotation (±15°)
        - Horizontal and vertical flip
        - Brightness/contrast adjustment
        - Gaussian blur
        - Optional: elastic transform, coarse dropout
        - Normalize with ImageNet stats
        - Convert to PyTorch tensor

    Args:
        img_size: Target image size (height, width).
        use_advanced: Whether to include elastic transform and coarse dropout.

    Returns:
        Albumentations Compose pipeline.
    """
    transforms_list = [
        # Resize
        A.Resize(height=img_size[0], width=img_size[1]),

        # Geometric transforms
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            p=0.3,
            border_mode=0,
        ),

        # Photometric transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 3), p=0.3),
        A.GaussNoise(p=0.2),
    ]

    # Advanced augmentation for small dataset
    if use_advanced:
        transforms_list.extend([
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=20,
                max_width=20,
                fill_value=0,
                p=0.3,
            ),
        ])

    # Normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    pipeline = A.Compose(transforms_list)
    logger.debug(
        f"Training augmentation pipeline created: {len(transforms_list)} transforms, "
        f"img_size={img_size}, advanced={use_advanced}"
    )
    return pipeline


def get_val_augmentation(
    img_size: tuple[int, int] = IMAGE_SIZE_CLASSIFICATION,
) -> A.Compose:
    """
    Get validation/test augmentation pipeline (resize + normalize only).

    Args:
        img_size: Target image size (height, width).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_detection_augmentation(
    img_size: tuple[int, int] = IMAGE_SIZE_DETECTION,
    train: bool = True,
) -> A.Compose:
    """
    Get augmentation pipeline for object detection (with bbox support).

    Args:
        img_size: Target image size (height, width).
        train: If True, includes augmentation; otherwise resize+normalize only.

    Returns:
        Albumentations Compose pipeline with bbox support.
    """
    if train:
        transforms_list = [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Rotate(limit=15, p=0.5, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        transforms_list = [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]

    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format="pascal_voc",  # [x_min, y_min, x_max, y_max]
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_tta_augmentations(
    img_size: tuple[int, int] = IMAGE_SIZE_CLASSIFICATION,
) -> list[A.Compose]:
    """
    Get Test-Time Augmentation (TTA) pipelines.

    Returns multiple augmentation pipelines that can be applied to a single
    image to generate multiple predictions for ensembling.

    Args:
        img_size: Target image size (height, width).

    Returns:
        List of Albumentations Compose pipelines.
    """
    base = [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    pipelines = [
        # Original
        A.Compose(base.copy()),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base.copy()),
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0)] + base.copy()),
        # Rotation +90
        A.Compose([A.Rotate(limit=(90, 90), p=1.0, border_mode=0)] + base.copy()),
        # Rotation -90
        A.Compose([A.Rotate(limit=(-90, -90), p=1.0, border_mode=0)] + base.copy()),
    ]

    return pipelines
