"""
Streamlit Web Application - Industrial Surface Defect Detection Platform.

Main entry point for the multi-page Streamlit application providing:
    - Image upload and defect detection
    - Analytics dashboard
    - Inspection history
    - Image gallery

Usage:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Steel Defect Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏭 Steel Defect Detection")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.markdown("""
- 📸 **Inspection** - Upload and analyze
- 📊 **Dashboard** - Analytics overview
- 📋 **History** - Past inspections
- 🖼️ **Gallery** - Image gallery
""")
st.sidebar.markdown("---")
st.sidebar.info("v1.0.0 | NEU Surface Defect Dataset")

# Main page content
st.title("🏭 Industrial Surface Defect Detection Platform")
st.markdown("""
Welcome to the **Steel Surface Defect Detection Platform**. This system uses 
deep learning to detect and classify defects in steel surfaces.

### Defect Classes Detected
| Class | Description |
|-------|-------------|
| Crazing | Fine crack patterns on the surface |
| Inclusion | Foreign particles embedded in steel |
| Patches | Irregular surface patches |
| Pitted Surface | Small pits/holes on the surface |
| Rolled-in Scale | Oxide scale pressed into the surface |
| Scratches | Linear marks from mechanical contact |

### Getting Started
1. Navigate to **Inspection** page to upload steel surface images
2. View detection results with bounding boxes and confidence scores
3. Check the **Dashboard** for quality-control analytics
4. Browse **History** for past inspection records

---
*Powered by PyTorch, YOLOv8, and FastAPI*
""")
