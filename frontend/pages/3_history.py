"""
Inspection History Page - View and manage past inspections.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="History", page_icon="📋", layout="wide")
st.title("📋 Inspection History")


def load_history(page: int = 1, page_size: int = 20, defect_filter=None):
    """Load inspection history from database."""
    try:
        from api.database.session import SessionLocal, init_db
        from api.database.crud import get_inspections, get_total_count

        init_db()
        db = SessionLocal()
        skip = (page - 1) * page_size
        inspections = get_inspections(
            db, skip=skip, limit=page_size, defect_type=defect_filter
        )
        total = get_total_count(db)
        db.close()
        return inspections, total
    except Exception:
        return [], 0


# Filters
col1, col2 = st.columns([2, 1])
with col1:
    defect_filter = st.selectbox(
        "Filter by Defect Type",
        ["All"] + [
            "Crazing", "Inclusion", "Patches",
            "Pitted_Surface", "Rolled-in_Scale", "Scratches"
        ],
    )
with col2:
    page_size = st.selectbox("Items per page", [10, 20, 50], index=1)

if defect_filter == "All":
    defect_filter = None

# Load data
inspections, total = load_history(page=1, page_size=page_size, defect_filter=defect_filter)

if inspections:
    # Convert to DataFrame
    data = []
    for insp in inspections:
        data.append({
            "ID": insp.id,
            "Image": insp.image_filename,
            "Defect Type": insp.defect_type,
            "Confidence": f"{insp.confidence:.2%}",
            "Inference (ms)": f"{insp.inference_time_ms:.1f}" if insp.inference_time_ms else "N/A",
            "Date": insp.created_at.strftime("%Y-%m-%d %H:%M") if insp.created_at else "N/A",
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(inspections)} of {total} records")

    # Export
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        "inspection_history.csv",
        "text/csv",
    )
else:
    st.info("No inspection records found. Run some inspections first.")
