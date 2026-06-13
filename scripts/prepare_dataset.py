"""
Prepare the NEU Surface Defect Dataset for training.

This script:
    1. Splits the dataset into train/val/test (70/15/15) with stratification
    2. Copies images into the processed directory structure
    3. Creates YOLO-format labels for object detection
    4. Generates a YOLO data.yaml configuration file

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CLASS_NAMES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    YOLO_DATA_DIR,
)
from src.data.annotation_converter import create_yolo_labels_from_classification
from src.data.dataset import NEUDataset
from src.data.preprocessing import split_dataset_indices
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_classification_split(
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Split dataset into train/val/test and copy images to processed directory.

    Creates the following structure:
        data/processed/
        ├── train/
        │   ├── Crazing/
        │   ├── Inclusion/
        │   └── ...
        ├── val/
        │   ├── Crazing/
        │   └── ...
        └── test/
            ├── Crazing/
            └── ...
    """
    logger.info("Preparing classification dataset split...")

    # Load dataset
    images_dir = RAW_DATA_DIR / "images"
    dataset = NEUDataset(root_dir=images_dir)

    if len(dataset) == 0:
        logger.error(
            f"No images found in {images_dir}. "
            "Run 'python scripts/download_dataset.py' first."
        )
        sys.exit(1)

    # Get labels for stratification
    labels = dataset.get_all_labels()
    paths = dataset.get_all_paths()

    # Split indices
    train_idx, val_idx, test_idx = split_dataset_indices(
        labels, train_ratio, val_ratio, test_ratio, seed
    )

    # Create directory structure and copy images
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split_name, indices in splits.items():
        split_dir = PROCESSED_DATA_DIR / split_name

        # Create class subdirectories
        for class_name in CLASS_NAMES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Copy images
        for idx in indices:
            src_path = paths[idx]
            class_name = dataset.get_class_name(idx)
            dst_path = split_dir / class_name / src_path.name
            shutil.copy2(str(src_path), str(dst_path))

        logger.info(f"  {split_name}: {len(indices)} images copied")

    # Print class distribution per split
    for split_name, indices in splits.items():
        class_counts = {}
        for idx in indices:
            cn = dataset.get_class_name(idx)
            class_counts[cn] = class_counts.get(cn, 0) + 1
        logger.info(f"  {split_name} distribution: {class_counts}")

    logger.info(f"Classification data prepared at: {PROCESSED_DATA_DIR}")


def prepare_yolo_dataset() -> None:
    """
    Prepare YOLO-format dataset from the processed classification split.

    Creates the following structure:
        data/yolo/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml
    """
    logger.info("Preparing YOLO dataset...")

    for split in ["train", "val", "test"]:
        split_img_dir = PROCESSED_DATA_DIR / split
        if not split_img_dir.exists():
            logger.error(
                f"Processed split not found: {split_img_dir}. "
                "Run classification split first."
            )
            return

        # Create YOLO structure
        yolo_img_dir = YOLO_DATA_DIR / "images" / split
        yolo_lbl_dir = YOLO_DATA_DIR / "labels" / split
        yolo_img_dir.mkdir(parents=True, exist_ok=True)
        yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to flat YOLO directory and create labels
        image_extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        count = 0

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = split_img_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in image_extensions:
                    continue

                # Copy image
                dst_img = yolo_img_dir / img_path.name
                shutil.copy2(str(img_path), str(dst_img))

                # Create YOLO label (full-image bbox)
                label_file = yolo_lbl_dir / f"{img_path.stem}.txt"
                with open(label_file, "w") as f:
                    f.write(f"{class_idx} 0.500000 0.500000 1.000000 1.000000\n")

                count += 1

        logger.info(f"  YOLO {split}: {count} images + labels")

    # Create data.yaml
    data_yaml_path = YOLO_DATA_DIR / "data.yaml"
    yaml_content = f"""# NEU Surface Defect Dataset - YOLO Configuration
# Auto-generated by scripts/prepare_dataset.py

path: {YOLO_DATA_DIR.resolve()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(CLASS_NAMES)}

# Class names
names: {CLASS_NAMES}
"""
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    logger.info(f"YOLO data.yaml created at: {data_yaml_path}")
    logger.info(f"YOLO dataset prepared at: {YOLO_DATA_DIR}")


def main() -> None:
    """Main entry point for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare NEU dataset for training")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val-ratio", type=float, default=VAL_SPLIT)
    parser.add_argument("--test-ratio", type=float, default=TEST_SPLIT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits")
    args = parser.parse_args()

    # Check if already prepared
    if not args.force and (PROCESSED_DATA_DIR / "train").exists():
        logger.info("Dataset already prepared. Use --force to re-split.")
        print("Dataset already prepared. Use --force to overwrite.")
        if not args.skip_yolo and not (YOLO_DATA_DIR / "data.yaml").exists():
            prepare_yolo_dataset()
        return

    # Clean existing processed data if forcing
    if args.force:
        if PROCESSED_DATA_DIR.exists():
            shutil.rmtree(str(PROCESSED_DATA_DIR))
        if YOLO_DATA_DIR.exists():
            shutil.rmtree(str(YOLO_DATA_DIR))

    # Prepare splits
    if not args.skip_classification:
        prepare_classification_split(
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )

    if not args.skip_yolo:
        prepare_yolo_dataset()

    print("\n✓ Dataset preparation complete!")
    print(f"  Classification data: {PROCESSED_DATA_DIR}")
    print(f"  YOLO data: {YOLO_DATA_DIR}")


if __name__ == "__main__":
    main()
