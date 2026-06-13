"""
Grad-CAM and feature map visualization for model explainability.

Generates attention heatmaps showing which image regions influenced
the model's prediction. Critical for quality-control validation.

Usage:
    from src.explainability.gradcam import GradCAMExplainer

    explainer = GradCAMExplainer(model)
    heatmap, overlay = explainer.explain(image_tensor)
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAMExplainer:
    """
    Grad-CAM visualization for CNN classification models.

    Generates class activation maps highlighting regions that contributed
    most to the model's prediction.

    Args:
        model: Trained classification model (DefectClassifier).
        target_layer: The CNN layer to compute Grad-CAM for.
                     If None, automatically selects the last conv layer.
        device: Device for computation.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Auto-detect target layer if not specified
        if target_layer is None:
            self.target_layer = self._find_last_conv_layer()
        else:
            self.target_layer = target_layer

        # Storage for gradients and activations
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

        logger.info(
            f"GradCAMExplainer initialized, target_layer: "
            f"{self.target_layer.__class__.__name__}"
        )

    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                if isinstance(module, nn.Conv2d):
                    last_conv = module

        if last_conv is None:
            # Try to get from backbone
            if hasattr(self.model, "features"):
                for module in self.model.features.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module

        if last_conv is None:
            raise ValueError("Could not find a convolutional layer in the model.")

        return last_conv

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for an input image.

        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W) or (C, H, W).
            target_class: Class to explain. If None, uses the predicted class.

        Returns:
            Tuple of (heatmap, predicted_class, confidence):
                - heatmap: Normalized heatmap array (H, W) in [0, 1].
                - predicted_class: The class index the model predicted.
                - confidence: Softmax confidence for the predicted class.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)
        probs = F.softmax(output, dim=1)

        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, (w, act) in enumerate(zip(weights, activations)):
            cam += w * act

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cam.cpu().numpy()

        return heatmap, target_class, confidence

    def generate_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            image: Original image (H, W) grayscale or (H, W, 3) RGB.
            heatmap: Grad-CAM heatmap (arbitrary size, will be resized).
            alpha: Blending factor (0=image only, 1=heatmap only).
            colormap: OpenCV colormap for heatmap visualization.

        Returns:
            Overlaid image (H, W, 3) in RGB format.
        """
        # Ensure image is RGB
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()

        h, w = image_rgb.shape[:2]

        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (
            (1 - alpha) * image_rgb.astype(np.float32)
            + alpha * heatmap_colored.astype(np.float32)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay


class FeatureMapVisualizer:
    """
    Visualize intermediate feature maps from CNN layers.

    Useful for understanding what patterns each layer has learned to detect.

    Args:
        model: Trained model.
        device: Device for computation.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.feature_maps = {}

    def extract_feature_maps(
        self,
        image_tensor: torch.Tensor,
        layer_names: Optional[list[str]] = None,
        max_layers: int = 5,
    ) -> dict[str, np.ndarray]:
        """
        Extract feature maps from specified layers.

        Args:
            image_tensor: Input image tensor (1, C, H, W) or (C, H, W).
            layer_names: Specific layer names to extract. If None, extracts
                        from all Conv2d layers (up to max_layers).
            max_layers: Maximum number of layers to extract from.

        Returns:
            Dictionary mapping layer names to feature map arrays.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        feature_maps = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                feature_maps[name] = output.detach().cpu().numpy()[0]
            return hook_fn

        # Register hooks on target layers
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if layer_names is None or name in layer_names:
                    hooks.append(module.register_forward_hook(make_hook(name)))
                    count += 1
                    if layer_names is None and count >= max_layers:
                        break

        # Forward pass
        with torch.no_grad():
            self.model(image_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return feature_maps

    @staticmethod
    def visualize_feature_map(
        feature_map: np.ndarray,
        num_channels: int = 16,
        figsize: tuple[int, int] = (12, 8),
    ):
        """
        Visualize channels of a feature map as a grid.

        Args:
            feature_map: Feature map array (C, H, W).
            num_channels: Number of channels to display.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        n_show = min(num_channels, feature_map.shape[0])
        cols = 4
        rows = (n_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for i in range(len(axes)):
            if i < n_show:
                axes[i].imshow(feature_map[i], cmap="viridis")
                axes[i].set_title(f"Channel {i}", fontsize=8)
            axes[i].axis("off")

        plt.suptitle("Feature Map Channels", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
