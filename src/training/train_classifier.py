"""
Training pipeline for defect classification models.

Complete training loop with validation, metric logging, checkpointing,
and early stopping. Supports mixed precision training and learning rate
scheduling.

Usage:
    from src.training.train_classifier import ClassifierTrainer

    trainer = ClassifierTrainer(model, train_loader, val_loader, config)
    history = trainer.train()
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from config.settings import DEVICE, MODELS_DIR, NUM_EPOCHS, LEARNING_RATE
from src.evaluation.metrics import compute_classification_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss, 'max' for accuracy/F1.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs without improvement."
                )

        return self.should_stop


class ClassifierTrainer:
    """
    Training orchestrator for defect classification models.

    Handles the complete training lifecycle including:
        - Training and validation loops
        - Mixed precision training (AMP)
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        - Metric logging

    Args:
        model: DefectClassifier model instance.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Training configuration dictionary.
        device: Device to train on ('cuda' or 'cpu').
        save_dir: Directory to save checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[dict] = None,
        device: str = DEVICE,
        save_dir: Optional[str | Path] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else MODELS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Default config
        self.config = {
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "scheduler": "cosine_annealing",
            "patience": 15,
            "use_amp": torch.cuda.is_available(),
            "unfreeze_epoch": 10,
            "gradient_clip": 1.0,
        }
        if config:
            self.config.update(config)

        # Setup optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(
            patience=self.config["patience"], mode="max"
        )

        # Mixed precision
        self.use_amp = self.config["use_amp"] and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "learning_rate": [],
        }

        logger.info(
            f"ClassifierTrainer initialized: device={device}, "
            f"epochs={self.config['epochs']}, lr={self.config['learning_rate']}, "
            f"AMP={self.use_amp}"
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        opt_name = self.config["optimizer"].lower()
        lr = self.config["learning_rate"]
        wd = self.config["weight_decay"]

        if opt_name == "adam":
            return Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return AdamW(params, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        else:
            return Adam(params, lr=lr, weight_decay=wd)

    def _build_scheduler(self):
        """Build learning rate scheduler from config."""
        sched_name = self.config.get("scheduler", "cosine_annealing")

        if sched_name == "cosine_annealing":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"],
                eta_min=1e-6,
            )
        elif sched_name == "step_lr":
            return StepLR(self.optimizer, step_size=15, gamma=0.1)
        else:
            return None

    def train(self) -> dict:
        """
        Execute the full training loop.

        Returns:
            Training history dictionary with loss, accuracy, and F1 per epoch.
        """
        best_f1 = 0.0
        epochs = self.config["epochs"]
        unfreeze_epoch = self.config.get("unfreeze_epoch", 0)

        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Unfreeze backbone at specified epoch
            if epoch == unfreeze_epoch and hasattr(self.model, "unfreeze_backbone"):
                logger.info(f"Epoch {epoch}: Unfreezing backbone for fine-tuning")
                self.model.unfreeze_backbone()
                # Rebuild optimizer with all parameters
                self.optimizer = self._build_optimizer()
                self.scheduler = self._build_scheduler()

            # Train one epoch
            train_loss, train_acc = self._train_epoch()

            # Validate
            val_loss, val_metrics = self._validate_epoch()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_macro"])
            self.history["learning_rate"].append(current_lr)

            # Log epoch results
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f} | LR: {current_lr:.6f}"
            )

            # Save best model
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            # Early stopping check
            if self.early_stopping(val_metrics["f1_macro"]):
                logger.info(f"Training stopped at epoch {epoch} (early stopping)")
                break

        elapsed = time.time() - start_time
        logger.info(
            f"Training complete! Best Val F1: {best_f1:.4f} | "
            f"Total time: {elapsed/60:.1f} minutes"
        )

        return self.history

    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip"]
                )
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self) -> tuple[float, dict]:
        """Run validation epoch. Returns (avg_loss, metrics_dict)."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(all_labels)
        metrics = compute_classification_metrics(
            np.array(all_labels), np.array(all_preds)
        )

        return avg_loss, metrics

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "backbone": getattr(self.model, "backbone_name", "unknown"),
        }

        if is_best:
            path = self.save_dir / "best_classifier.pth"
        else:
            path = self.save_dir / f"classifier_epoch_{epoch}.pth"

        torch.save(checkpoint, str(path))
        logger.debug(f"Checkpoint saved: {path}")
