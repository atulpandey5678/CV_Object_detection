"""
Visualization utilities for the Industrial Surface Defect Detection Platform.

Provides plotting functions for dataset exploration, training metrics,
detection results, and model explainability outputs.

Usage:
    from src.utils.visualization import (
        plot_class_distribution,
        show_images_grid,
        plot_training_history,
        draw_bounding_boxes,
    )
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config.settings import CLASS_NAMES, OUTPUTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style defaults
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_class_distribution(
    class_counts: dict[str, int],
    title: str = "NEU Dataset - Class Distribution",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot class distribution as a bar chart.

    Args:
        class_counts: Dictionary mapping class names to image counts.
        title: Plot title.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = sns.color_palette("husl", len(classes))

    # Bar chart
    bars = axes[0].bar(classes, counts, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Defect Class")
    axes[0].set_ylabel("Number of Images")
    axes[0].set_title(title)
    axes[0].tick_params(axis="x", rotation=45)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Pie chart
    axes[1].pie(
        counts,
        labels=classes,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Class Proportion")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved class distribution plot to: {save_path}")

    if show:
        plt.show()

    return fig


def show_images_grid(
    images: list[np.ndarray],
    titles: Optional[list[str]] = None,
    cols: int = 6,
    figsize: Optional[tuple[int, int]] = None,
    cmap: str = "gray",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """
    Display a grid of images with optional titles.

    Args:
        images: List of numpy arrays (images).
        titles: Optional list of titles for each image.
        cols: Number of columns in the grid.
        figsize: Figure size tuple (width, height).
        cmap: Colormap for grayscale images.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
        suptitle: Optional super title for the figure.

    Returns:
        Matplotlib Figure object.
    """
    n = len(images)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap=cmap)
            elif img.shape[2] == 1:
                ax.imshow(img.squeeze(), cmap=cmap)
            else:
                ax.imshow(img)

            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved image grid to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_sample_inspection(
    image: np.ndarray,
    class_name: str,
    confidence: Optional[float] = None,
    bboxes: Optional[list[list[float]]] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Display a single image with inspection results.

    Args:
        image: Input image (grayscale or RGB).
        class_name: Predicted/true class name.
        confidence: Optional confidence score (0-1).
        bboxes: Optional list of bounding boxes [x1, y1, x2, y2].
        save_path: Optional path to save the figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Draw bounding boxes if provided
    display_img = image.copy()
    if bboxes:
        if display_img.ndim == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if display_img.ndim == 2:
        ax.imshow(display_img, cmap="gray")
    else:
        ax.imshow(display_img)

    title = f"Defect: {class_name}"
    if confidence is not None:
        title += f" (conf: {confidence:.2%})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss',
                 'train_acc', 'val_acc', etc.
        title: Plot title.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib Figure object.
    """
    # Determine which metrics are available
    loss_keys = [k for k in history if "loss" in k.lower()]
    acc_keys = [k for k in history if "acc" in k.lower() or "f1" in k.lower()]

    num_plots = sum([bool(loss_keys), bool(acc_keys)])
    if num_plots == 0:
        num_plots = 1

    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Loss plot
    if loss_keys:
        for key in loss_keys:
            axes[plot_idx].plot(history[key], label=key, linewidth=2)
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Loss")
        axes[plot_idx].set_title("Loss")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    # Accuracy/F1 plot
    if acc_keys:
        for key in acc_keys:
            axes[plot_idx].plot(history[key], label=key, linewidth=2)
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Score")
        axes[plot_idx].set_title("Accuracy / F1")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved training history plot to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (num_classes × num_classes).
        class_names: List of class names for axis labels.
        title: Plot title.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
        normalize: Whether to normalize values to percentages.

    Returns:
        Matplotlib Figure object.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        square=True,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to: {save_path}")

    if show:
        plt.show()

    return fig


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: list[list[float]],
    labels: list[str],
    scores: Optional[list[float]] = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes with labels on an image.

    Args:
        image: Input image (BGR or RGB format).
        boxes: List of bounding boxes [x1, y1, x2, y2].
        labels: List of class name labels for each box.
        scores: Optional list of confidence scores.
        color: Box color in BGR format.
        thickness: Line thickness.

    Returns:
        Image with drawn bounding boxes.
    """
    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    # Color palette for different classes
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (0, 128, 128), (128, 128, 0),
    ]

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(c) for c in box[:4]]
        box_color = palette[i % len(palette)]

        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, thickness)

        # Build label text
        text = label
        if scores and i < len(scores):
            text += f" {scores[i]:.2f}"

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            output,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            box_color,
            -1,
        )

        # Draw label text
        cv2.putText(
            output,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def plot_pixel_intensity_distribution(
    images: list[np.ndarray],
    labels: list[str],
    title: str = "Pixel Intensity Distribution by Class",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot pixel intensity histograms for different classes.

    Args:
        images: List of grayscale images.
        labels: Class name for each image.
        title: Plot title.
        save_path: Optional save path.
        show: Whether to display.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    unique_labels = sorted(set(labels))
    colors = sns.color_palette("husl", len(unique_labels))

    for class_name, color in zip(unique_labels, colors):
        class_images = [img for img, lbl in zip(images, labels) if lbl == class_name]
        all_pixels = np.concatenate([img.flatten() for img in class_images])
        ax.hist(
            all_pixels, bins=50, alpha=0.5, color=color,
            label=class_name, density=True,
        )

    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
