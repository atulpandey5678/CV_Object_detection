"""
Production inference pipeline for steel surface defect detection.

Provides a unified interface for single image, batch, and folder inference
with both classification and detection capabilities.

Usage:
    from src.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(classifier_path="models/best_classifier.pth")
    result = pipeline.predict_single("path/to/image.jpg")
    # {"defect_type": "Scratches", "confidence": 0.95, "bounding_boxes": [...]}
"""

import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.settings import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    IMAGE_SIZE_CLASSIFICATION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
)
from src.data.preprocessing import preprocess_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """
    Production-ready inference pipeline for defect detection.

    Supports:
        - Single image inference
        - Batch inference
        - Folder inference
        - Classification-only or detection+classification modes

    Args:
        classifier_path: Path to classification model weights.
        detector_path: Optional path to detection model weights.
        classifier_backbone: Backbone architecture used for classifier.
        detector_type: Detection model type ('yolov8' or 'faster_rcnn').
        confidence_threshold: Minimum confidence for predictions.
        device: Device for inference ('cuda' or 'cpu').
    """

    def __init__(
        self,
        classifier_path: Optional[str | Path] = None,
        detector_path: Optional[str | Path] = None,
        classifier_backbone: str = "resnet50",
        detector_type: str = "yolov8",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: str = DEVICE,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.class_names = CLASS_NAMES
        self.classifier = None
        self.detector = None

        # Load classifier
        if classifier_path:
            self._load_classifier(classifier_path, classifier_backbone)

        # Load detector
        if detector_path:
            self._load_detector(detector_path, detector_type)

        if not self.classifier and not self.detector:
            logger.warning(
                "No models loaded. Provide classifier_path or detector_path."
            )

    def _load_classifier(self, path: str | Path, backbone: str) -> None:
        """Load classification model."""
        from src.models.classifier import DefectClassifier

        try:
            self.classifier = DefectClassifier.load_from_checkpoint(
                str(path), backbone=backbone, device=self.device
            )
            logger.info(f"Classifier loaded: {path}")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")

    def _load_detector(self, path: str | Path, model_type: str) -> None:
        """Load detection model."""
        from src.models.detector import DefectDetector

        try:
            self.detector = DefectDetector(
                model_type=model_type,
                weights_path=str(path),
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
            logger.info(f"Detector loaded: {path}")
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")

    def predict_single(
        self,
        image_source: Union[str, Path, np.ndarray],
        mode: str = "auto",
    ) -> dict:
        """
        Run inference on a single image.

        Args:
            image_source: Image file path or numpy array.
            mode: 'classification', 'detection', or 'auto'.
                  'auto' uses detection if available, else classification.

        Returns:
            Prediction result dictionary:
            {
                "defect_type": str,
                "confidence": float,
                "bounding_boxes": [[x1, y1, x2, y2], ...],
                "all_scores": {class_name: confidence, ...},
                "inference_time_ms": float,
                "image_path": str | None,
            }
        """
        start_time = time.time()

        # Load image
        image = self._load_image(image_source)
        if image is None:
            return self._empty_result("Failed to load image")

        # Determine mode
        if mode == "auto":
            mode = "detection" if self.detector else "classification"

        # Run inference
        if mode == "detection" and self.detector:
            result = self._run_detection(image)
        elif self.classifier:
            result = self._run_classification(image)
        else:
            return self._empty_result("No model available for inference")

        # Add metadata
        elapsed_ms = (time.time() - start_time) * 1000
        result["inference_time_ms"] = round(elapsed_ms, 2)
        result["image_path"] = str(image_source) if isinstance(image_source, (str, Path)) else None

        return result

    def predict_batch(
        self,
        image_sources: list[Union[str, Path, np.ndarray]],
        mode: str = "auto",
    ) -> list[dict]:
        """
        Run inference on a batch of images.

        Args:
            image_sources: List of image paths or numpy arrays.
            mode: Inference mode.

        Returns:
            List of prediction result dictionaries.
        """
        results = []
        start_time = time.time()

        if mode == "auto":
            mode = "detection" if self.detector else "classification"

        if mode == "classification" and self.classifier:
            # Batch classification
            results = self._batch_classify(image_sources)
        else:
            # Sequential for detection (YOLO handles batching internally)
            for source in image_sources:
                results.append(self.predict_single(source, mode=mode))

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Batch inference: {len(image_sources)} images in {elapsed_ms:.1f}ms "
            f"({elapsed_ms/len(image_sources):.1f}ms/image)"
        )

        return results

    def predict_folder(
        self,
        folder_path: str | Path,
        mode: str = "auto",
        extensions: set[str] = None,
    ) -> list[dict]:
        """
        Run inference on all images in a folder.

        Args:
            folder_path: Path to folder containing images.
            mode: Inference mode.
            extensions: Set of valid image extensions.

        Returns:
            List of prediction result dictionaries.
        """
        folder_path = Path(folder_path)
        if extensions is None:
            extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

        image_paths = sorted([
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ])

        if not image_paths:
            logger.warning(f"No images found in: {folder_path}")
            return []

        logger.info(f"Processing {len(image_paths)} images from: {folder_path}")
        return self.predict_batch(image_paths, mode=mode)

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from path or return numpy array directly."""
        if isinstance(source, np.ndarray):
            return source

        path = Path(source)
        if not path.exists():
            logger.error(f"Image not found: {path}")
            return None

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Failed to read image: {path}")
            return None

        return image

    @torch.no_grad()
    def _run_classification(self, image: np.ndarray) -> dict:
        """Run classification on a single image."""
        # Preprocess
        tensor = preprocess_image(
            image,
            target_size=IMAGE_SIZE_CLASSIFICATION,
            normalize=True,
            to_tensor=True,
        )
        tensor = tensor.unsqueeze(0).to(self.device)

        # Predict
        output = self.classifier(tensor)
        probs = F.softmax(output, dim=1)[0]

        # Get top prediction
        confidence, class_idx = probs.max(0)
        defect_type = self.class_names[class_idx.item()]

        # All class scores
        all_scores = {
            name: float(probs[i])
            for i, name in enumerate(self.class_names)
        }

        return {
            "defect_type": defect_type,
            "confidence": float(confidence),
            "bounding_boxes": [],
            "all_scores": all_scores,
            "class_index": int(class_idx),
        }

    def _run_detection(self, image: np.ndarray) -> dict:
        """Run detection on a single image."""
        # Convert grayscale to RGB for detector
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        detections = self.detector.predict(image_rgb)

        if not detections or not detections[0]["boxes"]:
            return {
                "defect_type": "No Defect Detected",
                "confidence": 0.0,
                "bounding_boxes": [],
                "all_scores": {},
                "class_index": -1,
            }

        det = detections[0]
        # Use highest confidence detection as primary result
        max_idx = np.argmax(det["scores"])

        return {
            "defect_type": det["class_names"][max_idx],
            "confidence": float(det["scores"][max_idx]),
            "bounding_boxes": det["boxes"],
            "all_scores": {
                name: float(score)
                for name, score in zip(det["class_names"], det["scores"])
            },
            "class_index": int(det["labels"][max_idx]),
            "all_detections": det,
        }

    @torch.no_grad()
    def _batch_classify(self, image_sources: list) -> list[dict]:
        """Batch classification inference."""
        results = []
        batch_tensors = []
        valid_indices = []

        for i, source in enumerate(image_sources):
            image = self._load_image(source)
            if image is None:
                results.append(self._empty_result("Failed to load image"))
                continue

            tensor = preprocess_image(
                image,
                target_size=IMAGE_SIZE_CLASSIFICATION,
                normalize=True,
                to_tensor=True,
            )
            batch_tensors.append(tensor)
            valid_indices.append(i)

        if not batch_tensors:
            return results

        # Stack and predict
        batch = torch.stack(batch_tensors).to(self.device)
        outputs = self.classifier(batch)
        probs_batch = F.softmax(outputs, dim=1)

        # Build results
        result_map = {}
        for idx, probs in zip(valid_indices, probs_batch):
            confidence, class_idx = probs.max(0)
            all_scores = {
                name: float(probs[i])
                for i, name in enumerate(self.class_names)
            }
            result_map[idx] = {
                "defect_type": self.class_names[class_idx.item()],
                "confidence": float(confidence),
                "bounding_boxes": [],
                "all_scores": all_scores,
                "class_index": int(class_idx),
                "image_path": str(image_sources[idx]) if isinstance(image_sources[idx], (str, Path)) else None,
                "inference_time_ms": 0,
            }

        # Assemble final list in order
        final_results = []
        for i in range(len(image_sources)):
            if i in result_map:
                final_results.append(result_map[i])
            else:
                final_results.append(self._empty_result("Failed to load image"))

        return final_results

    @staticmethod
    def _empty_result(message: str) -> dict:
        """Return an empty result with error message."""
        return {
            "defect_type": "Error",
            "confidence": 0.0,
            "bounding_boxes": [],
            "all_scores": {},
            "class_index": -1,
            "error": message,
            "inference_time_ms": 0,
            "image_path": None,
        }
