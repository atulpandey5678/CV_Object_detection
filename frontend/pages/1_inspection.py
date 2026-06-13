"""
Inspection Page - Upload images and run defect detection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Inspection", page_icon="📸", layout="wide")
st.title("📸 Defect Inspection")
st.markdown("Upload a steel surface image to detect defects.")


@st.cache_resource
def load_pipeline():
    """Load the inference pipeline (cached)."""
    from src.inference.pipeline import InferencePipeline
    from config.settings import MODELS_DIR

    classifier_path = MODELS_DIR / "best_classifier.pth"
    detector_path = MODELS_DIR / "best_detector.pt"

    cls_path = str(classifier_path) if classifier_path.exists() else None
    det_path = str(detector_path) if detector_path.exists() else None

    return InferencePipeline(
        classifier_path=cls_path,
        detector_path=det_path,
    )


def draw_results(image: np.ndarray, result: dict) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    display_img = image.copy()
    if display_img.ndim == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)

    boxes = result.get("bounding_boxes", [])
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box[:4]]
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return display_img


# File upload
uploaded_files = st.file_uploader(
    "Upload steel surface image(s)",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

if uploaded_files:
    pipeline = load_pipeline()

    for uploaded_file in uploaded_files:
        st.markdown("---")
        col1, col2 = st.columns(2)

        # Read image
        file_bytes = uploaded_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            st.error(f"Failed to load: {uploaded_file.name}")
            continue

        # Display original
        with col1:
            st.subheader("Original Image")
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        # Run inference
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            result = pipeline.predict_single(image)

        # Display results
        with col2:
            st.subheader("Detection Results")

            # Draw bounding boxes
            result_img = draw_results(image, result)
            st.image(result_img, caption="Detection Output", use_container_width=True)

        # Metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Defect Type", result["defect_type"])
        with col_b:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        with col_c:
            st.metric("Inference Time", f"{result.get('inference_time_ms', 0):.1f} ms")

        # Detailed scores
        if result.get("all_scores"):
            with st.expander("All Class Scores"):
                for class_name, score in sorted(
                    result["all_scores"].items(), key=lambda x: x[1], reverse=True
                ):
                    st.progress(score, text=f"{class_name}: {score:.4f}")

        # Save to database
        try:
            from api.database.session import SessionLocal, init_db
            from api.database.crud import create_inspection

            init_db()
            db = SessionLocal()
            create_inspection(
                db=db,
                image_path=uploaded_file.name,
                image_filename=uploaded_file.name,
                defect_type=result["defect_type"],
                confidence=result["confidence"],
                bounding_boxes=result.get("bounding_boxes"),
                all_scores=result.get("all_scores"),
                inference_time_ms=result.get("inference_time_ms"),
            )
            db.close()
        except Exception:
            pass  # DB save is optional
else:
    st.info("👆 Upload one or more steel surface images to start inspection.")
