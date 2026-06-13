"""
Annotation format converter for object detection tasks.

Converts between VOC XML, YOLO TXT, and COCO JSON annotation formats.
This is essential for training different detection architectures
(YOLO requires TXT, Faster R-CNN uses COCO/VOC format).

Usage:
    from src.data.annotation_converter import (
        voc_to_yolo,
        yolo_to_voc,
        create_yolo_labels_from_classification,
    )
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import CLASS_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def voc_to_yolo(
    xml_path: str | Path,
    class_names: list[str] = CLASS_NAMES,
) -> list[str]:
    """
    Convert a VOC XML annotation to YOLO format.

    VOC format: (xmin, ymin, xmax, ymax) in absolute pixels
    YOLO format: (class_id, x_center, y_center, width, height) normalized [0, 1]

    Args:
        xml_path: Path to the VOC XML annotation file.
        class_names: List of class names for index mapping.

    Returns:
        List of YOLO-format annotation lines.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_names:
            logger.warning(f"Unknown class '{class_name}' in {xml_path}")
            continue

        class_id = class_names.index(class_name)
        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (normalized center + width/height)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_lines


def yolo_to_voc(
    txt_path: str | Path,
    img_width: int,
    img_height: int,
    class_names: list[str] = CLASS_NAMES,
) -> list[dict]:
    """
    Convert YOLO TXT annotations to VOC-style bounding boxes.

    Args:
        txt_path: Path to the YOLO annotation text file.
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        class_names: List of class names.

    Returns:
        List of dicts with keys: class_name, xmin, ymin, xmax, ymax.
    """
    txt_path = Path(txt_path)
    annotations = []

    if not txt_path.exists():
        return annotations

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert to absolute pixel coordinates
            xmin = (x_center - width / 2) * img_width
            ymin = (y_center - height / 2) * img_height
            xmax = (x_center + width / 2) * img_width
            ymax = (y_center + height / 2) * img_height

            annotations.append({
                "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                "class_id": class_id,
                "xmin": int(xmin),
                "ymin": int(ymin),
                "xmax": int(xmax),
                "ymax": int(ymax),
            })

    return annotations


def create_yolo_labels_from_classification(
    images_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str] = CLASS_NAMES,
    full_image_bbox: bool = True,
) -> int:
    """
    Generate YOLO labels from a classification-organized dataset.

    Since the NEU dataset is organized by class folders but may not have
    per-object bounding box annotations, this creates YOLO labels where
    the entire image is treated as the bounding box for the defect.

    Args:
        images_dir: Path to class-organized images directory.
        output_dir: Path to write YOLO label files.
        class_names: List of class names.
        full_image_bbox: If True, use full image as bounding box.

    Returns:
        Number of label files created.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    image_extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    for class_idx, class_name in enumerate(class_names):
        class_dir = images_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in image_extensions:
                continue

            label_file = output_dir / f"{img_path.stem}.txt"

            if full_image_bbox:
                # Full image bounding box: center=0.5, size=1.0
                line = f"{class_idx} 0.500000 0.500000 1.000000 1.000000"
            else:
                # Read image to get actual defect region (could use edge detection)
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Default to full image
                line = f"{class_idx} 0.500000 0.500000 1.000000 1.000000"

            with open(label_file, "w") as f:
                f.write(line + "\n")
            count += 1

    logger.info(f"Created {count} YOLO label files in: {output_dir}")
    return count


def create_coco_json(
    images_dir: str | Path,
    output_path: str | Path,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """
    Create a COCO-format JSON annotation file from class-organized images.

    Args:
        images_dir: Path to class-organized images directory.
        output_path: Path to write the COCO JSON file.
        class_names: List of class names.

    Returns:
        The COCO annotation dictionary.
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Categories (COCO is 1-indexed)
    for idx, name in enumerate(class_names):
        coco["categories"].append({
            "id": idx + 1,
            "name": name,
            "supercategory": "defect",
        })

    image_id = 0
    ann_id = 0
    image_extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    for class_idx, class_name in enumerate(class_names):
        class_dir = images_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in image_extensions:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape[:2]
            image_id += 1

            coco["images"].append({
                "id": image_id,
                "file_name": f"{class_name}/{img_path.name}",
                "height": h,
                "width": w,
            })

            # Full-image bounding box annotation
            ann_id += 1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_idx + 1,
                "bbox": [0, 0, w, h],  # COCO bbox: [x, y, width, height]
                "area": w * h,
                "iscrowd": 0,
            })

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        f"Created COCO JSON with {len(coco['images'])} images and "
        f"{len(coco['annotations'])} annotations at: {output_path}"
    )
    return coco
