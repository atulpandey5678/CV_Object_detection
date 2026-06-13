"""
Defect Detection Models for the NEU Surface Defect Dataset.

Supports YOLOv8 (via Ultralytics) and Faster R-CNN (via torchvision).
Provides a unified interface for training, inference, and export.

Usage:
    from src.models.detector import DefectDetector

    detector = DefectDetector(model_type="yolov8", model_size="s")
    results = detector.predict("path/to/image.jpg")
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config.settings import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    IOU_THRESHOLD,
    MODELS_DIR,
    NUM_CLASSES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DefectDetector:
    """
    Unified defect detection interface supporting YOLOv8 and Faster R-CNN.

    Args:
        model_type: 'yolov8' or 'faster_rcnn'.
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x').
        num_classes: Number of defect classes.
        confidence_threshold: Minimum confidence for predictions.
        iou_threshold: IoU threshold for NMS.
        device: Device for inference.
        weights_path: Optional path to pretrained weights.
    """

    def __init__(
        self,
        model_type: str = "yolov8",
        model_size: str = "s",
        num_classes: int = NUM_CLASSES,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: str = DEVICE,
        weights_path: Optional[str | Path] = None,
    ):
        self.model_type = model_type
        self.model_size = model_size
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None

        if weights_path:
            self.load(weights_path)
        else:
            self._build_model()

        logger.info(
            f"DefectDetector({model_type}, size={model_size}): "
            f"classes={num_classes}, device={device}"
        )

    def _build_model(self) -> None:
        """Build the detection model from config."""
        if self.model_type == "yolov8":
            self._build_yolov8()
        elif self.model_type == "faster_rcnn":
            self._build_faster_rcnn()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _build_yolov8(self) -> None:
        """Build YOLOv8 model using Ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package required for YOLOv8. "
                "Install with: pip install ultralytics"
            )

        model_name = f"yolov8{self.model_size}.pt"
        self.model = YOLO(model_name)
        logger.info(f"YOLOv8{self.model_size} model built (pretrained COCO weights)")

    def _build_faster_rcnn(self) -> None:
        """Build Faster R-CNN with ResNet50-FPN backbone."""
        # num_classes + 1 for background
        self.model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

        # Replace the classification head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes + 1  # +1 for background
        )

        self.model.to(self.device)
        logger.info("Faster R-CNN (ResNet50-FPN v2) model built")

    def train_yolo(
        self,
        data_yaml: str | Path,
        epochs: int = 100,
        img_size: int = 640,
        batch_size: int = 16,
        project: str = "runs/detect",
        name: str = "neu_defect",
        **kwargs,
    ) -> dict:
        """
        Train YOLOv8 model on the defect dataset.

        Args:
            data_yaml: Path to YOLO data.yaml configuration file.
            epochs: Number of training epochs.
            img_size: Input image size.
            batch_size: Training batch size.
            project: Project directory for saving results.
            name: Experiment name.
            **kwargs: Additional YOLO training parameters.

        Returns:
            Training results dictionary.
        """
        if self.model_type != "yolov8":
            raise ValueError("train_yolo() only works with YOLOv8 models")

        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project=project,
            name=name,
            device=self.device if self.device != "cpu" else "cpu",
            patience=20,
            save=True,
            verbose=True,
            **kwargs,
        )

        logger.info(f"YOLOv8 training complete. Results saved to: {project}/{name}")
        return results

    def predict(
        self,
        source: str | Path | np.ndarray,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> list[dict]:
        """
        Run detection on an image or batch of images.

        Args:
            source: Image path, numpy array, or directory path.
            confidence: Override default confidence threshold.
            iou: Override default IoU threshold.

        Returns:
            List of detection results, each containing:
                - boxes: [[x1, y1, x2, y2], ...]
                - scores: [conf1, conf2, ...]
                - labels: [class_id1, ...]
                - class_names: [name1, ...]
        """
        conf = confidence or self.confidence_threshold
        iou_thresh = iou or self.iou_threshold

        if self.model_type == "yolov8":
            return self._predict_yolo(source, conf, iou_thresh)
        elif self.model_type == "faster_rcnn":
            return self._predict_faster_rcnn(source, conf, iou_thresh)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _predict_yolo(
        self, source, conf: float, iou: float
    ) -> list[dict]:
        """Run YOLOv8 inference."""
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device if self.device != "cpu" else "cpu",
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            detection = {
                "boxes": boxes.xyxy.cpu().numpy().tolist() if len(boxes) > 0 else [],
                "scores": boxes.conf.cpu().numpy().tolist() if len(boxes) > 0 else [],
                "labels": boxes.cls.cpu().numpy().astype(int).tolist() if len(boxes) > 0 else [],
                "class_names": [
                    CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"class_{int(c)}"
                    for c in boxes.cls.cpu().numpy()
                ] if len(boxes) > 0 else [],
            }
            detections.append(detection)

        return detections

    @torch.no_grad()
    def _predict_faster_rcnn(
        self, source, conf: float, iou: float
    ) -> list[dict]:
        """Run Faster R-CNN inference."""
        import cv2

        self.model.eval()

        # Load image
        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = source

        # Preprocess
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Predict
        predictions = self.model(img_tensor)[0]

        # Filter by confidence
        keep = predictions["scores"] > conf
        boxes = predictions["boxes"][keep].cpu().numpy().tolist()
        scores = predictions["scores"][keep].cpu().numpy().tolist()
        labels = predictions["labels"][keep].cpu().numpy().astype(int).tolist()

        # Labels from Faster R-CNN are 1-indexed (0=background)
        labels = [l - 1 for l in labels]
        class_names = [
            CLASS_NAMES[l] if 0 <= l < len(CLASS_NAMES) else f"class_{l}"
            for l in labels
        ]

        return [{
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "class_names": class_names,
        }]

    def export_onnx(self, output_path: Optional[str | Path] = None) -> Path:
        """
        Export model to ONNX format.

        Args:
            output_path: Path for the exported model.

        Returns:
            Path to the exported ONNX file.
        """
        if output_path is None:
            output_path = MODELS_DIR / f"defect_detector_{self.model_type}.onnx"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_type == "yolov8":
            self.model.export(format="onnx", imgsz=640)
            logger.info(f"YOLOv8 exported to ONNX")
        else:
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                opset_version=11,
                input_names=["input"],
                output_names=["boxes", "labels", "scores"],
            )
            logger.info(f"Faster R-CNN exported to ONNX: {output_path}")

        return output_path

    def load(self, weights_path: str | Path) -> None:
        """Load model weights from a file."""
        weights_path = Path(weights_path)

        if self.model_type == "yolov8":
            from ultralytics import YOLO
            self.model = YOLO(str(weights_path))
        elif self.model_type == "faster_rcnn":
            self._build_faster_rcnn()
            state_dict = torch.load(str(weights_path), map_location=self.device)
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict)
            self.model.eval()

        logger.info(f"Loaded weights from: {weights_path}")

    def validate(self, data_yaml: Optional[str | Path] = None) -> dict:
        """
        Validate the detection model.

        Args:
            data_yaml: Path to YOLO data.yaml (for YOLO models).

        Returns:
            Validation metrics dictionary.
        """
        if self.model_type == "yolov8" and data_yaml:
            metrics = self.model.val(data=str(data_yaml))
            return {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            }
        else:
            logger.warning("Validation requires YOLO model with data.yaml")
            return {}
