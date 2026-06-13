"""
Evaluation metrics for classification and detection models.

Provides functions to compute accuracy, precision, recall, F1-score,
confusion matrix, and detection metrics (IoU, mAP).

Usage:
    from src.evaluation.metrics import compute_classification_metrics
    metrics = compute_classification_metrics(y_true, y_pred)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import CLASS_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (integer array).
        y_pred: Predicted labels (integer array).
        class_names: List of class names for per-class metrics.

    Returns:
        Dictionary with accuracy, precision, recall, F1 (macro and per-class),
        confusion matrix, and classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        if i < len(precision_per_class):
            per_class[name] = {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class": per_class,
    }


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box [x1, y1, x2, y2].
        box2: Second box [x1, y1, x2, y2].

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Compute Average Precision (AP) using the 11-point interpolation method.

    Args:
        precisions: Array of precision values at each threshold.
        recalls: Array of recall values at each threshold.

    Returns:
        Average Precision value.
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += prec_at_recall.max()
    ap /= 11.0
    return ap


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    num_classes: int = 6,
) -> dict:
    """
    Compute mean Average Precision (mAP) for object detection.

    Args:
        predictions: List of prediction dicts with keys:
            - boxes: [[x1, y1, x2, y2], ...]
            - scores: [conf1, conf2, ...]
            - labels: [class_id1, class_id2, ...]
            - image_id: identifier
        ground_truths: List of ground truth dicts with keys:
            - boxes: [[x1, y1, x2, y2], ...]
            - labels: [class_id1, class_id2, ...]
            - image_id: identifier
        iou_threshold: IoU threshold for true positive.
        num_classes: Number of classes.

    Returns:
        Dictionary with mAP, per-class AP, precision, and recall.
    """
    aps = []

    for class_id in range(num_classes):
        # Collect all predictions and GTs for this class
        class_preds = []
        class_gts = {}
        n_gt = 0

        for gt in ground_truths:
            img_id = gt["image_id"]
            gt_boxes = [
                box for box, lbl in zip(gt["boxes"], gt["labels"])
                if lbl == class_id
            ]
            class_gts[img_id] = {
                "boxes": gt_boxes,
                "matched": [False] * len(gt_boxes),
            }
            n_gt += len(gt_boxes)

        for pred in predictions:
            img_id = pred["image_id"]
            for box, score, lbl in zip(pred["boxes"], pred["scores"], pred["labels"]):
                if lbl == class_id:
                    class_preds.append({
                        "image_id": img_id,
                        "box": box,
                        "score": score,
                    })

        if n_gt == 0:
            continue

        # Sort predictions by confidence (descending)
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            img_id = pred["image_id"]
            pred_box = np.array(pred["box"])

            if img_id not in class_gts:
                fp[i] = 1
                continue

            gt_data = class_gts[img_id]
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_data["boxes"]):
                iou = compute_iou(pred_box, np.array(gt_box))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and not gt_data["matched"][best_gt_idx]:
                tp[i] = 1
                gt_data["matched"][best_gt_idx] = True
            else:
                fp[i] = 1

        # Compute precision/recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / n_gt

        ap = compute_ap(precisions, recalls)
        aps.append(ap)

    mean_ap = np.mean(aps) if aps else 0.0

    return {
        "mAP": float(mean_ap),
        "per_class_AP": {CLASS_NAMES[i]: float(ap) for i, ap in enumerate(aps)},
        "iou_threshold": iou_threshold,
    }
