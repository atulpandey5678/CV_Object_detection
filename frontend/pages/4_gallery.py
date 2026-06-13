"""
Image Gallery Page - Browse past inspection images.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Gallery", page_icon="🖼️", layout="wide")
st.title("🖼️ Inspection Gallery")


def load_gallery_items(limit: int = 24):
    """Load recent inspection images from database."""
    try:
        from api.database.session import SessionLocal, init_db
        from api.database.crud import get_inspections

        init_db()
        db = SessionLocal()
        inspections = get_inspections(db, limit=limit)
        db.close()
        return inspections
    except Exception:
        return []


inspections = load_gallery_items()

if inspections:
    cols_per_row = 4
    for i in range(0, len(inspections), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(inspections):
                break

            insp = inspections[idx]
            with col:
                # Try to load image
                img_path = Path(insp.image_path)
                if img_path.exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        st.image(img, caption=insp.image_filename, use_container_width=True)
                else:
                    st.text(f"📄 {insp.image_filename}")

                # Label
                confidence_color = "🟢" if insp.confidence > 0.8 else "🟡" if insp.confidence > 0.5 else "🔴"
                st.caption(
                    f"{confidence_color} {insp.defect_type} ({insp.confidence:.0%})"
                )
else:
    st.info("No images in gallery. Upload and inspect images first.")
