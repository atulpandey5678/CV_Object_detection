"""
Defect Classification Models using Transfer Learning.

Supports ResNet50, EfficientNet-B0, and MobileNetV2 backbones with
configurable fine-tuning strategies.

Usage:
    from src.models.classifier import DefectClassifier

    model = DefectClassifier(backbone="resnet50", num_classes=6, pretrained=True)
    output = model(images)
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from config.settings import NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported backbone architectures
SUPPORTED_BACKBONES = ["resnet50", "efficientnet_b0", "mobilenet_v2"]


class DefectClassifier(nn.Module):
    """
    Multi-class defect classifier using pretrained CNN backbones.

    Replaces the final classification head with a custom head suitable
    for the NEU 6-class problem. Supports layer freezing for transfer learning.

    Args:
        backbone: Architecture name ('resnet50', 'efficientnet_b0', 'mobilenet_v2').
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained weights.
        dropout: Dropout rate before final classification layer.
        freeze_backbone: Whether to freeze backbone layers initially.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from: {SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Build backbone and classification head
        self.features, in_features = self._build_backbone(backbone, pretrained)
        self.classifier = self._build_head(in_features, num_classes, dropout)

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"DefectClassifier({backbone}): {total_params:,} params, "
            f"{trainable_params:,} trainable, classes={num_classes}"
        )

    def _build_backbone(
        self, backbone: str, pretrained: bool
    ) -> tuple[nn.Module, int]:
        """Build feature extractor from pretrained backbone."""
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet50":
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            # Remove original classifier
            model.fc = nn.Identity()
            return model, in_features

        elif backbone == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, in_features

        elif backbone == "mobilenet_v2":
            model = models.mobilenet_v2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, in_features

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _build_head(
        self, in_features: int, num_classes: int, dropout: float
    ) -> nn.Sequential:
        """Build custom classification head."""
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        features = self.features(x)
        if features.dim() > 2:
            features = features.flatten(1)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers (only classifier head is trainable)."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info(f"Backbone frozen: {self.backbone_name}")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Backbone unfrozen: {trainable:,} trainable parameters")

    def unfreeze_last_n_layers(self, n: int) -> None:
        """
        Unfreeze the last N layers of the backbone.

        Useful for gradual unfreezing during fine-tuning.

        Args:
            n: Number of layers to unfreeze from the end.
        """
        layers = list(self.features.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"Unfroze last {n} layers of {self.backbone_name}")

    def get_feature_extractor(self) -> nn.Module:
        """Return the backbone feature extractor (for Grad-CAM etc.)."""
        return self.features

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        backbone: str = "resnet50",
        num_classes: int = NUM_CLASSES,
        device: str = "cpu",
    ) -> "DefectClassifier":
        """
        Load a model from a saved checkpoint.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            backbone: Architecture name used during training.
            num_classes: Number of output classes.
            device: Device to load the model on.

        Returns:
            Loaded DefectClassifier model in eval mode.
        """
        model = cls(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle checkpoint with extra metadata
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        logger.info(f"Loaded classifier from: {checkpoint_path}")
        return model
