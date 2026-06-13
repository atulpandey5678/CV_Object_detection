"""
Analytics Dashboard - Quality control metrics and trends.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Quality Control Dashboard")


def get_analytics_data() -> dict:
    """Fetch analytics from the database."""
    try:
        from api.database.session import SessionLocal, init_db
        from api.database.crud import get_analytics

        init_db()
        db = SessionLocal()
        data = get_analytics(db)
        db.close()
        return data
    except Exception as e:
        return {
            "total_inspections": 0,
            "defect_counts": {},
            "avg_confidence": 0.0,
            "avg_inference_time_ms": 0.0,
            "most_common_defect": "N/A",
            "daily_counts": [],
        }


# Fetch data
analytics = get_analytics_data()

# KPI Cards
st.markdown("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Inspections", analytics["total_inspections"])
with col2:
    st.metric("Avg Confidence", f"{analytics['avg_confidence']:.1%}")
with col3:
    st.metric("Most Common Defect", analytics["most_common_defect"])
with col4:
    st.metric("Avg Inference Time", f"{analytics['avg_inference_time_ms']:.1f} ms")

st.markdown("---")

# Charts
if analytics["defect_counts"]:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Defect Distribution")
        df_defects = pd.DataFrame(
            list(analytics["defect_counts"].items()),
            columns=["Defect Type", "Count"],
        )
        fig_pie = px.pie(
            df_defects,
            values="Count",
            names="Defect Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Defect Counts")
        fig_bar = px.bar(
            df_defects,
            x="Defect Type",
            y="Count",
            color="Defect Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Daily trend
    if analytics["daily_counts"]:
        st.subheader("Inspection Trends")
        df_daily = pd.DataFrame(analytics["daily_counts"])
        df_daily["date"] = pd.to_datetime(df_daily["date"])
        fig_trend = px.line(
            df_daily,
            x="date",
            y="count",
            title="Daily Inspections",
            markers=True,
        )
        fig_trend.update_layout(height=350)
        st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No inspection data yet. Run some inspections to see analytics.")
