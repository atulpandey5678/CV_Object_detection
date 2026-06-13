"""
Main training entry point for defect classification models.

Supports training ResNet50, EfficientNet-B0, and MobileNetV2 with
configurable hyperparameters, MLflow tracking, and model checkpointing.

Usage:
    python scripts/train.py --backbone resnet50 --epochs 50 --lr 0.0001
    python scripts/train.py --backbone efficientnet_b0 --batch-size 16
    python scripts/train.py --compare-all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from config.settings import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_EPOCHS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from src.data.augmentation import get_train_augmentation, get_val_augmentation
from src.data.dataset import NEUDataset
from src.models.classifier import DefectClassifier
from src.training.train_classifier import ClassifierTrainer
from src.utils.logger import get_logger
from src.utils.visualization import plot_training_history, plot_confusion_matrix

logger = get_logger(__name__)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders from processed dataset."""
    train_transform = get_train_augmentation()
    val_transform = get_val_augmentation()

    train_dataset = NEUDataset(
        root_dir=PROCESSED_DATA_DIR / "train",
        transform=train_transform,
    )
    val_dataset = NEUDataset(
        root_dir=PROCESSED_DATA_DIR / "val",
        transform=val_transform,
    )
    test_dataset = NEUDataset(
        root_dir=PROCESSED_DATA_DIR / "test",
        transform=val_transform,
    )

    # Adjust num_workers for Windows
    if sys.platform == "win32":
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def train_single_model(
    backbone: str,
    epochs: int,
    lr: float,
    batch_size: int,
    freeze: bool = True,
    unfreeze_epoch: int = 10,
) -> dict:
    """Train a single classification model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {backbone}")
    logger.info(f"{'='*60}")

    set_seed()

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(batch_size)

    # Create model
    model = DefectClassifier(
        backbone=backbone,
        pretrained=True,
        freeze_backbone=freeze,
        dropout=0.5,
    )

    # Training config
    config = {
        "epochs": epochs,
        "learning_rate": lr,
        "optimizer": "adam",
        "weight_decay": 0.0001,
        "scheduler": "cosine_annealing",
        "patience": 15,
        "use_amp": torch.cuda.is_available(),
        "unfreeze_epoch": unfreeze_epoch,
        "gradient_clip": 1.0,
    }

    # Save directory
    save_dir = MODELS_DIR / backbone
    save_dir.mkdir(parents=True, exist_ok=True)

    # MLflow tracking
    try:
        from mlops.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker("defect-classification")
        tracker.start_run(run_name=f"{backbone}_training")
        tracker.log_params({
            "backbone": backbone,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "freeze_backbone": freeze,
            "unfreeze_epoch": unfreeze_epoch,
        })
    except Exception:
        tracker = None

    # Train
    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=DEVICE,
        save_dir=save_dir,
    )

    history = trainer.train()

    # Log final metrics to MLflow
    if tracker:
        final_metrics = {
            "final_val_acc": history["val_acc"][-1],
            "final_val_f1": history["val_f1"][-1],
            "best_val_f1": max(history["val_f1"]),
        }
        tracker.log_metrics(final_metrics)
        tracker.log_training_history(history)
        tracker.end_run()

    # Save training history plot
    plot_training_history(
        history,
        title=f"{backbone} Training History",
        save_path=save_dir / "training_history.png",
        show=False,
    )

    logger.info(f"{backbone} training complete. Best Val F1: {max(history['val_f1']):.4f}")
    return history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train defect classifier")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "mobilenet_v2"],
        help="Model backbone architecture",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--unfreeze-epoch", type=int, default=10)
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Train and compare all backbones",
    )

    args = parser.parse_args()

    # Ensure processed data exists
    if not (PROCESSED_DATA_DIR / "train").exists():
        logger.error(
            "Processed dataset not found. Run 'python scripts/prepare_dataset.py' first."
        )
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare_all:
        # Train all backbones for comparison
        results = {}
        for backbone in ["resnet50", "efficientnet_b0", "mobilenet_v2"]:
            history = train_single_model(
                backbone=backbone,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                freeze=not args.no_freeze,
                unfreeze_epoch=args.unfreeze_epoch,
            )
            results[backbone] = max(history["val_f1"])

        # Print comparison
        print("\n" + "=" * 50)
        print("Model Comparison Results")
        print("=" * 50)
        for backbone, best_f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {backbone:<20} Best F1: {best_f1:.4f}")
        print("=" * 50)
    else:
        train_single_model(
            backbone=args.backbone,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            freeze=not args.no_freeze,
            unfreeze_epoch=args.unfreeze_epoch,
        )


if __name__ == "__main__":
    main()
