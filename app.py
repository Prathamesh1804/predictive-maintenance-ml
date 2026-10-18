"""
Predictive Maintenance Intelligence System — Streamlit Dashboard

Pages:
  A. Executive Overview
  B. Exploratory Data Analysis
  C. Predict Machine Failure
  D. Model Evaluation
  E. Batch Prediction
  F. Model Explainability
  G. About the Project

Run:
    streamlit run app.py
"""
import sys
import io
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.config import (
    BASE_FEATURES, CATEGORICAL_FEATURES, NUMERIC_BASE_FEATURES,
    TARGET_COLUMN, LEAKAGE_COLUMNS, ID_COLUMNS,
    PRODUCT_TYPE_OPTIONS, FEATURE_RANGES, DEFAULT_THRESHOLD,
    MODEL_REGISTRY_FILE, COMPARISON_FILE, METRICS_FILE,
    RAW_DIR, MODELS_DIR,
)
from src.data_loader import find_raw_csv, load_and_validate
from src.predict import get_prediction_service, list_available_models, get_best_model_name
from src.utils import load_json

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Intelligence System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — industrial analytics theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark industrial palette */
:root {
    --bg-primary: #0d1117;
    --bg-card: #161b22;
    --bg-card2: #1c2128;
    --accent-blue: #2f81f7;
    --accent-green: #3fb950;
    --accent-orange: #d29922;
    --accent-red: #f85149;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border: #30363d;
}

.main { background-color: var(--bg-primary); }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2f81f7;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.8rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Page header */
.page-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d2040 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.page-header h1 {
    color: #e6edf3;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
}
.page-header p {
    color: #8b949e;
    font-size: 0.95rem;
    margin: 0;
}

/* Prediction result */
.result-failure {
    background: linear-gradient(135deg, #2d1116 0%, #3d1520 100%);
    border: 2px solid #f85149;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #0d2016 0%, #0f2d1c 100%);
    border: 2px solid #3fb950;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}
.result-prob {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* Alert boxes */
.info-box {
    background: #0d2040;
    border-left: 4px solid #2f81f7;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #a5c8f0;
}
.warning-box {
    background: #2d2011;
    border-left: 4px solid #d29922;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #e3b757;
}

/* Tables */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2f81f7, #1a6fd4);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(47,129,247,0.4);
}

.synthetic-notice {
    background: linear-gradient(135deg, #1a1400 0%, #2d2011 100%);
    border: 1px solid #d29922;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #e3b757;
    margin: 0.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_data():
    """Load and validate the raw dataset."""
    try:
        df, report = load_and_validate()
        return df, report, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner=False)
def cached_prediction_service(model_name=None):
    return get_prediction_service(model_name)


@st.cache_data(show_spinner=False)
def cached_load_registry():
    if MODEL_REGISTRY_FILE.exists():
        return load_json(MODEL_REGISTRY_FILE)
    return None


@st.cache_data(show_spinner=False)
def cached_load_comparison():
    if COMPARISON_FILE.exists():
        return pd.read_csv(COMPARISON_FILE)
    return None


@st.cache_data(show_spinner=False)
def cached_load_metrics():
    if METRICS_FILE.exists():
        return load_json(METRICS_FILE)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "#161b22",
    "plot_bgcolor": "#0d1117",
    "font": {"family": "Inter", "color": "#e6edf3"},
}

MODEL_COLORS = {
    "Logistic Regression": "#2f81f7",
    "Random Forest":       "#3fb950",
    "SVM":                 "#f85149",
    "Gradient Boosting":   "#d29922",
}


def apply_theme(fig, title=None):
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=PLOTLY_THEME["font"],
        title_font_size=15,
        title_font_color="#e6edf3",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
    )
    if title:
        fig.update_layout(title_text=title)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 1.5rem;'>
            <div style='font-size:2.5rem'>⚙️</div>
            <div style='font-size:1.05rem; font-weight:700; color:#e6edf3; margin-top:0.5rem;'>
                PredictiveMaint.AI
            </div>
            <div style='font-size:0.75rem; color:#8b949e; margin-top:0.2rem;'>
                Intelligent Failure Detection
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        page = st.radio(
            "Navigation",
            options=[
                "🏠 Executive Overview",
                "📊 Exploratory Data Analysis",
                "🔮 Predict Machine Failure",
                "📈 Model Evaluation",
                "📦 Batch Prediction",
                "🧠 Model Explainability",
                "ℹ️ About the Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Model selector
        available = list_available_models()
        if available:
            best = get_best_model_name() or available[0]
            selected_model = st.selectbox(
                "Active Model",
                options=available,
                index=available.index(best) if best in available else 0,
                help="Select which trained model to use for predictions.",
            )
        else:
            selected_model = None
            st.caption("⚠️ No trained models found.")

        # Threshold slider
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.1, max_value=0.9,
            value=DEFAULT_THRESHOLD, step=0.05,
            help="Adjust the probability threshold for classifying a machine as failing. "
                 "Lower values → more failures detected (higher recall, lower precision).",
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem; color:#8b949e;'>"
            "Dataset: UCI AI4I 2020<br>"
            "⚠️ Synthetic dataset — academic use only<br>"
            "Not a certified safety system."
            "</div>",
            unsafe_allow_html=True,
        )

    return page, selected_model, threshold


# ─────────────────────────────────────────────────────────────────────────────
# PAGE A: Executive Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(df, report, registry, selected_model, threshold):
    st.markdown("""
    <div class='page-header'>
        <h1>⚙️ Executive Overview</h1>
        <p>High-level summary of the dataset and selected model performance.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='synthetic-notice'>
        ⚠️ <strong>Synthetic Dataset Notice:</strong> The UCI AI4I 2020 dataset is artificially generated
        to model industrial conditions. It does not originate from real machine telemetry.
        This system is an academic prototype only — not a certified industrial safety tool.
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not loaded. See sidebar for instructions.")
        return

    # ── KPI row ──────────────────────────────────────────────────────────────
    total = len(df)
    failures = int(df[TARGET_COLUMN].sum())
    no_failures = total - failures
    failure_rate = failures / total * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{total:,}", "Total Records"),
        (c2, f"{failures:,}", "Failure Events"),
        (c3, f"{no_failures:,}", "Normal Operations"),
        (c4, f"{failure_rate:.2f}%", "Failure Rate"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model metrics ─────────────────────────────────────────────────────────
    comparison = cached_load_comparison()
    if comparison is not None and selected_model:
        st.subheader("🏆 Selected Model Performance")
        row = comparison[comparison["Model"] == selected_model]
        if not row.empty:
            row = row.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, key, label in [
                (c1, "Precision", "Precision"),
                (c2, "Recall",    "Recall"),
                (c3, "F1-Score",  "F1-Score"),
                (c4, "ROC-AUC",   "ROC-AUC"),
                (c5, "PR-AUC",    "PR-AUC"),
            ]:
                col.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='font-size:1.6rem'>{row[key]:.4f}</div>
                    <div class='metric-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout: class distribution + failure modes ─────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Class Distribution")
        class_counts = df[TARGET_COLUMN].value_counts().rename(
            index={0: "No Failure", 1: "Failure"})
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            color=class_counts.index,
            color_discrete_map={"No Failure": "#3fb950", "Failure": "#f85149"},
            hole=0.5,
        )
        fig = apply_theme(fig)
        fig.update_traces(textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Failure Mode Breakdown")
        fm_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
        if fm_cols:
            fm_data = df[fm_cols].sum().sort_values(ascending=False)
            fig = px.bar(
                x=fm_data.index, y=fm_data.values,
                color=fm_data.index,
                labels={"x": "Failure Mode", "y": "Count"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig = apply_theme(fig, "Failure Mode Counts (Descriptive)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Key findings ──────────────────────────────────────────────────────────
    st.subheader("📌 Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-box'>
        <strong>Dataset Characteristics</strong><br>
        • 10,000 synthetic observations<br>
        • 6 input features (3 numeric operational, 1 categorical, 2 temperature)<br>
        • Highly imbalanced: ~3.4% failures<br>
        • No missing values in the official dataset
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-box'>
        <strong>ML Approach</strong><br>
        • 4 models trained and compared<br>
        • Optimised for recall (detecting failures is priority)<br>
        • Class weighting applied to handle imbalance<br>
        • Features engineered from physics-based relationships
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='warning-box'>
    <strong>⚠️ Important Disclaimer:</strong>
    Model predictions are probability estimates based on patterns in a synthetic dataset.
    They must not be used as the sole basis for safety-critical decisions.
    Always follow certified maintenance protocols and qualified engineering judgement.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE B: Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────

def page_eda(df):
    st.markdown("""
    <div class='page-header'>
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Understand patterns in the dataset through interactive visualizations.</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not loaded.")
        return

    # ── Dataset summary ───────────────────────────────────────────────────────
    with st.expander("📋 Dataset Summary", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns)}")
        c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        st.dataframe(df.describe().round(3), use_container_width=True)

    # ── Missing values ────────────────────────────────────────────────────────
    mv = df.isnull().sum()
    if mv.any():
        st.warning("⚠️ Missing values detected.")
        st.dataframe(mv[mv > 0].to_frame("Missing Count"))
    else:
        st.success("✅ No missing values in the dataset.")

    st.markdown("---")

    # ── Numerical distributions ───────────────────────────────────────────────
    st.subheader("📐 Numerical Feature Distributions")
    filter_type = st.selectbox(
        "Filter by Product Type (or All)",
        options=["All"] + PRODUCT_TYPE_OPTIONS,
        key="eda_type_filter",
    )
    df_filtered = df if filter_type == "All" else df[df["Type"] == filter_type]

    numeric_cols = NUMERIC_BASE_FEATURES
    fig = make_subplots(rows=2, cols=3, subplot_titles=numeric_cols,
                        vertical_spacing=0.15, horizontal_spacing=0.08)
    for i, col in enumerate(numeric_cols):
        r, c = divmod(i, 3)
        no_fail = df_filtered[df_filtered[TARGET_COLUMN] == 0][col]
        fail    = df_filtered[df_filtered[TARGET_COLUMN] == 1][col]
        fig.add_trace(go.Histogram(x=no_fail, name="No Failure", opacity=0.7,
                                   marker_color="#3fb950", showlegend=(i == 0)), row=r+1, col=c+1)
        fig.add_trace(go.Histogram(x=fail,    name="Failure",    opacity=0.8,
                                   marker_color="#f85149", showlegend=(i == 0)), row=r+1, col=c+1)
    fig = apply_theme(fig)
    fig.update_layout(barmode="overlay", height=500, legend_title="Class")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Correlation heatmap ────────────────────────────────────────────────────
    st.subheader("🔗 Correlation Heatmap")
    num_df = df[NUMERIC_BASE_FEATURES + [TARGET_COLUMN]].copy()
    corr = num_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig = apply_theme(fig, "Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
    <em>Correlation describes linear associations between variables. It does not imply causation.
    A high correlation with the failure target may indicate predictive value but should be
    interpreted in the context of the manufacturing process.</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Failure rate by product type ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏭 Failure Rate by Product Type")
        rate_by_type = (
            df.groupby("Type")[TARGET_COLUMN]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .rename(columns={"mean": "Failure Rate", "sum": "Failures", "count": "Total"})
        )
        rate_by_type["Failure Rate %"] = (rate_by_type["Failure Rate"] * 100).round(2)
        fig = px.bar(
            rate_by_type, x="Type", y="Failure Rate %",
            color="Type", text="Failure Rate %",
            color_discrete_map={"L": "#2f81f7", "M": "#d29922", "H": "#f85149"},
        )
        fig = apply_theme(fig)
        fig.update_layout(showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌡️ Temperature vs Rotational Speed")
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig = px.scatter(
            sample,
            x="Air temperature [K]",
            y="Rotational speed [rpm]",
            color=sample[TARGET_COLUMN].map({0: "No Failure", 1: "Failure"}),
            color_discrete_map={"No Failure": "#3fb950", "Failure": "#f85149"},
            opacity=0.6,
        )
        fig = apply_theme(fig, "Air Temperature vs Rotational Speed")
        st.plotly_chart(fig, use_container_width=True)

    # ── Torque vs Tool wear ───────────────────────────────────────────────────
    st.subheader("⚙️ Torque vs Tool Wear")
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="Tool wear [min]", y="Torque [Nm]",
        color=sample[TARGET_COLUMN].map({0: "No Failure", 1: "Failure"}),
        color_discrete_map={"No Failure": "#3fb950", "Failure": "#f85149"},
        opacity=0.6,
        marginal_x="histogram", marginal_y="histogram",
    )
    fig = apply_theme(fig, "Torque vs Tool Wear (coloured by outcome)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Failure modes (descriptive only) ──────────────────────────────────────
    st.subheader("🔍 Failure Mode Distribution (Descriptive)")
    st.markdown("""
    <div class='info-box'>
    The UCI dataset includes pre-labeled failure modes: TWF (Tool Wear Failure),
    HDF (Heat Dissipation Failure), PWF (Power Failure), OSF (Overstrain Failure),
    RNF (Random Failure). These are shown here for <em>exploratory analysis only</em>
    and are <strong>never</strong> used as input features for model training.
    </div>""", unsafe_allow_html=True)

    fm_cols = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
    if fm_cols:
        fm_df = df[fm_cols].sum().reset_index()
        fm_df.columns = ["Failure Mode", "Count"]
        fig = px.bar(
            fm_df, x="Failure Mode", y="Count",
            color="Failure Mode",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig = apply_theme(fig)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE C: Predict Machine Failure
# ─────────────────────────────────────────────────────────────────────────────

def page_predict(selected_model, threshold):
    st.markdown("""
    <div class='page-header'>
        <h1>🔮 Predict Machine Failure</h1>
        <p>Enter current machine operating parameters to assess failure risk.</p>
    </div>
    """, unsafe_allow_html=True)

    service = cached_prediction_service(selected_model)
    if not service.is_ready:
        st.error(
            "⚠️ No trained model found. Please run the training pipeline first:\n\n"
            "```powershell\npython scripts/run_pipeline.py\n```"
        )
        return

    st.markdown("""
    <div class='warning-box'>
    ⚠️ Predictions are probability estimates from a model trained on a <strong>synthetic</strong>
    dataset. Results should inform — not replace — qualified engineering inspection.
    A "No Failure" prediction does not guarantee a machine is safe.
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.subheader("🛠️ Machine Operating Parameters")

        c1, c2, c3 = st.columns(3)
        with c1:
            prod_type = st.selectbox(
                "Product Type",
                options=PRODUCT_TYPE_OPTIONS,
                index=1,
                help="L = Low quality, M = Medium quality, H = High quality",
            )
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=260.0, max_value=340.0,
                value=300.0, step=0.1,
                help=f"Typical range: {FEATURE_RANGES['Air temperature [K]']}",
            )
        with c2:
            proc_temp = st.number_input(
                "Process Temperature [K]",
                min_value=270.0, max_value=360.0,
                value=310.0, step=0.1,
                help=f"Typical range: {FEATURE_RANGES['Process temperature [K]']}",
            )
            rpm = st.number_input(
                "Rotational Speed [rpm]",
                min_value=0, max_value=5000,
                value=1500, step=10,
                help=f"Typical range: {FEATURE_RANGES['Rotational speed [rpm]']}",
            )
        with c3:
            torque = st.number_input(
                "Torque [Nm]",
                min_value=0.0, max_value=200.0,
                value=40.0, step=0.5,
                help=f"Typical range: {FEATURE_RANGES['Torque [Nm]']}",
            )
            tool_wear = st.number_input(
                "Tool Wear [min]",
                min_value=0, max_value=500,
                value=100, step=1,
                help=f"Typical range: {FEATURE_RANGES['Tool wear [min]']}",
            )

        # Show threshold info
        st.caption(f"Decision threshold: **{threshold:.2f}** — adjustable in the sidebar.")
        submitted = st.form_submit_button("⚡ Analyse Failure Risk", use_container_width=True)

    if not submitted:
        return

    # ── Validate inputs ───────────────────────────────────────────────────────
    warnings_list = []
    r = FEATURE_RANGES
    if not (r["Air temperature [K]"][0] <= air_temp <= r["Air temperature [K]"][1]):
        warnings_list.append(f"Air temperature {air_temp}K is outside typical range {r['Air temperature [K]']}")
    if not (r["Process temperature [K]"][0] <= proc_temp <= r["Process temperature [K]"][1]):
        warnings_list.append(f"Process temperature {proc_temp}K is outside typical range {r['Process temperature [K]']}")
    if proc_temp <= air_temp:
        warnings_list.append("Process temperature should generally exceed air temperature.")
    if not (r["Rotational speed [rpm]"][0] <= rpm <= r["Rotational speed [rpm]"][1]):
        warnings_list.append(f"Rotational speed {rpm}rpm is outside typical range {r['Rotational speed [rpm]']}")

    for w in warnings_list:
        st.warning(f"⚠️ {w}")

    # ── Run prediction ────────────────────────────────────────────────────────
    input_data = {
        "Type":                     prod_type,
        "Air temperature [K]":      air_temp,
        "Process temperature [K]":  proc_temp,
        "Rotational speed [rpm]":   rpm,
        "Torque [Nm]":              torque,
        "Tool wear [min]":          tool_wear,
    }

    try:
        result = service.predict_single(input_data, threshold=threshold)
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        return

    prob = result["failure_probability"]
    prob_pct = result["failure_probability_pct"]
    is_failure = result["predicted_label"] == 1

    # ── Result display ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    res_class = "result-failure" if is_failure else "result-safe"
    icon = "🔴" if is_failure else "🟢"
    label = result["predicted_class"]
    color = "#f85149" if is_failure else "#3fb950"

    st.markdown(f"""
    <div class='{res_class}'>
        <div class='result-prob' style='color:{color}'>{prob_pct:.1f}%</div>
        <div style='font-size:1.1rem; color:{color}; font-weight:600; margin-top:0.3rem;'>
            Estimated Failure Probability
        </div>
        <div class='result-title' style='color:{color}; margin-top:0.8rem;'>
            {icon} {label}
        </div>
        <div style='color:#8b949e; font-size:0.85rem; margin-top:0.5rem;'>
            Model: {result['model_name']} | Threshold: {threshold:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8b949e"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#0d2016"},
                {"range": [30, 60], "color": "#2d2011"},
                {"range": [60, 100], "color": "#2d1116"},
            ],
            "threshold": {
                "line": {"color": "#e6edf3", "width": 2},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
    ))
    fig = apply_theme(fig)
    fig.update_layout(height=300, margin=dict(t=20, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # Maintenance recommendation
    if prob_pct >= 70:
        rec = "🚨 **HIGH RISK** — Elevated predicted failure probability. Immediate inspection recommended."
    elif prob_pct >= threshold * 100:
        rec = "⚠️ **ELEVATED RISK** — Inspect the machine. Review operating parameters and maintenance records."
    elif prob_pct >= 20:
        rec = "🟡 **MODERATE** — Routine monitoring advised. No immediate action required based on this estimate."
    else:
        rec = "✅ **LOW RISK** — Operating parameters appear within normal bounds. Continue scheduled maintenance."

    st.info(rec)
    st.caption(
        "ℹ️ This recommendation is generated by a model trained on synthetic data. "
        "It does not replace professional engineering assessment or safety protocols."
    )

    # Threshold explanation
    with st.expander("ℹ️ About the Decision Threshold"):
        st.markdown(f"""
        The current threshold is **{threshold:.2f}**.

        - If the failure probability ≥ {threshold:.2f} → predicted **Failure**
        - If the failure probability < {threshold:.2f} → predicted **No Failure**

        **Trade-off:**
        - **Lower threshold** → More machines flagged as failing → Higher recall (fewer missed failures), but more false alarms.
        - **Higher threshold** → Fewer false alarms, but some real failures may be missed.

        In predictive maintenance, **missing a real failure is often more costly** than a false alarm,
        so a lower threshold is typically preferred. Adjust the threshold using the sidebar slider.
        """)

    # Feature values used
    with st.expander("📋 Feature Values Used for Prediction"):
        feat_df = pd.DataFrame([input_data]).T.rename(columns={0: "Value"})
        st.dataframe(feat_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE D: Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def page_evaluation():
    st.markdown("""
    <div class='page-header'>
        <h1>📈 Model Evaluation</h1>
        <p>Comprehensive comparison of all trained models on the held-out test set.</p>
    </div>
    """, unsafe_allow_html=True)

    comparison = cached_load_comparison()
    metrics = cached_load_metrics()
    registry = cached_load_registry()

    if comparison is None:
        st.warning(
            "No evaluation results found. Train the models first:\n"
            "```powershell\npython scripts/run_pipeline.py\n```"
        )
        return

    st.markdown("""
    <div class='info-box'>
    <strong>Why accuracy alone is misleading for imbalanced datasets:</strong><br>
    With only ~3.4% failures, a model that always predicts "No Failure" would achieve ~96.6% accuracy
    — yet it would miss every single failure. This system therefore prioritises
    <strong>Recall</strong> (detecting real failures) and <strong>PR-AUC</strong>
    (area under the Precision-Recall curve), which are far more informative
    for imbalanced binary classification.
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison table ───────────────────────────────────────────────────────
    st.subheader("📊 Model Comparison Table")
    best = registry["best_model"] if registry else None
    styled = comparison.style.highlight_max(
        subset=["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
        color="#0d2040",
    ).format(precision=4)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if best:
        st.success(f"⭐ Best model by PR-AUC: **{best}**")

    st.markdown("---")

    # ── ROC curves ────────────────────────────────────────────────────────────
    roc_path = ROOT / "reports" / "figures" / "roc_curves.png"
    pr_path  = ROOT / "reports" / "figures" / "pr_curves.png"
    cm_path  = ROOT / "reports" / "figures" / "confusion_matrices.png"
    comp_path = ROOT / "reports" / "figures" / "metric_comparison.png"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        if roc_path.exists():
            st.image(str(roc_path), use_column_width=True)
        else:
            st.caption("ROC curve plot not found. Run the training pipeline.")
    with col2:
        st.subheader("Precision-Recall Curves")
        if pr_path.exists():
            st.image(str(pr_path), use_column_width=True)
        else:
            st.caption("PR curve plot not found. Run the training pipeline.")

    st.subheader("Confusion Matrices")
    if cm_path.exists():
        st.image(str(cm_path), use_column_width=True)
    else:
        st.caption("Confusion matrix plot not found.")

    st.subheader("Metric Comparison")
    if comp_path.exists():
        st.image(str(comp_path), use_column_width=True)
    else:
        st.caption("Metric comparison plot not found.")

    # ── Per-model details ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Per-Model Detailed Metrics")
    if metrics:
        for model_name, m in metrics.get("models", {}).items():
            with st.expander(f"📋 {model_name}"):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                for col, key, label in [
                    (c1, "accuracy",  "Accuracy"),
                    (c2, "precision", "Precision"),
                    (c3, "recall",    "Recall"),
                    (c4, "f1",        "F1"),
                    (c5, "roc_auc",   "ROC-AUC"),
                    (c6, "pr_auc",    "PR-AUC"),
                ]:
                    col.metric(label, f"{m.get(key, 0):.4f}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("True Positives",  m.get("true_positives",  0))
                c2.metric("True Negatives",  m.get("true_negatives",  0))
                c3.metric("False Positives", m.get("false_positives", 0))
                c4.metric("False Negatives", m.get("false_negatives", 0))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE E: Batch Prediction
# ─────────────────────────────────────────────────────────────────────────────

def page_batch(selected_model, threshold):
    st.markdown("""
    <div class='page-header'>
        <h1>📦 Batch Prediction</h1>
        <p>Upload a CSV file with multiple machine observations for bulk failure-risk scoring.</p>
    </div>
    """, unsafe_allow_html=True)

    service = cached_prediction_service(selected_model)

    # ── Sample template ───────────────────────────────────────────────────────
    st.subheader("📥 Download Sample Input Template")
    sample_data = pd.DataFrame({
        "Type":                     ["L", "M", "H", "M", "L"],
        "Air temperature [K]":      [298.1, 300.5, 302.3, 299.8, 301.2],
        "Process temperature [K]":  [308.6, 310.1, 312.7, 309.3, 311.0],
        "Rotational speed [rpm]":   [1551, 1408, 1862, 2205, 1350],
        "Torque [Nm]":              [42.8, 46.3, 28.9, 20.1, 55.4],
        "Tool wear [min]":          [0, 53, 117, 198, 240],
    })
    st.dataframe(sample_data, use_container_width=True, hide_index=True)
    csv_bytes = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Template CSV",
        data=csv_bytes,
        file_name="batch_input_template.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <strong>Required columns (exact names):</strong><br>
    <code>Type</code>, <code>Air temperature [K]</code>, <code>Process temperature [K]</code>,
    <code>Rotational speed [rpm]</code>, <code>Torque [Nm]</code>, <code>Tool wear [min]</code>
    <br><br>
    Do NOT include failure labels or model output columns — they will be added automatically.
    </div>
    """, unsafe_allow_html=True)

    # ── File upload ────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload CSV for Batch Prediction",
        type=["csv"],
        help="Upload a CSV containing the six required input feature columns.",
    )

    if uploaded is None:
        return

    try:
        input_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.subheader(f"Preview — {len(input_df):,} rows uploaded")
    st.dataframe(input_df.head(10), use_container_width=True)

    # Validate columns
    ok, missing = service.validate_batch_columns(input_df)
    if not ok:
        st.error(f"❌ Missing required columns: **{missing}**\nPlease fix the CSV and re-upload.")
        return

    if not service.is_ready:
        st.error("No trained model found. Run the training pipeline first.")
        return

    # Run batch prediction
    with st.spinner("Running batch prediction..."):
        try:
            result_df = service.predict_batch(input_df, threshold=threshold)
        except Exception as exc:
            st.error(f"Batch prediction error: {exc}")
            return

    st.success(f"✅ Predictions complete for {len(result_df):,} records.")

    # ── Results summary ───────────────────────────────────────────────────────
    n_fail = int((result_df["predicted_label"] == 1).sum())
    n_safe = len(result_df) - n_fail
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(result_df):,}")
    c2.metric("Predicted Failures", f"{n_fail:,}")
    c3.metric("Predicted No Failure", f"{n_safe:,}")

    # Distribution
    fig = px.histogram(
        result_df, x="failure_probability_pct",
        color="predicted_class",
        color_discrete_map={"No Failure": "#3fb950", "Failure": "#f85149"},
        nbins=50,
        labels={"failure_probability_pct": "Failure Probability (%)"},
    )
    fig = apply_theme(fig, "Distribution of Predicted Failure Probabilities")
    st.plotly_chart(fig, use_container_width=True)

    # Result table
    st.subheader("📋 Prediction Results")
    display_cols = BASE_FEATURES + [
        "failure_probability_pct", "predicted_class", "model_name"
    ]
    display_cols = [c for c in display_cols if c in result_df.columns]
    st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

    # Download
    result_csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction Results CSV",
        data=result_csv,
        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE F: Model Explainability
# ─────────────────────────────────────────────────────────────────────────────

def page_explainability(df, selected_model):
    st.markdown("""
    <div class='page-header'>
        <h1>🧠 Model Explainability</h1>
        <p>Understand which features drive the model's failure predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    from src.explainability import (
        get_best_importance, get_tree_feature_importance,
        get_lr_coefficients, get_permutation_importance, get_shap_values
    )
    from src.predict import load_registry
    from sklearn.model_selection import train_test_split
    from config.config import RANDOM_SEED, TEST_SIZE

    registry = cached_load_registry()
    if registry is None:
        st.warning("No trained models found. Run the training pipeline first.")
        return

    if selected_model is None:
        selected_model = registry.get("best_model")

    model_info = registry["models"].get(selected_model)
    if model_info is None:
        st.error(f"Model '{selected_model}' not in registry.")
        return

    import joblib
    try:
        pipeline = joblib.load(model_info["path"])
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    st.success(f"Explaining: **{selected_model}**")

    # ── Feature importance ────────────────────────────────────────────────────
    st.subheader("📊 Global Feature Importance")

    imp = get_best_importance(pipeline)
    if "error" in imp:
        st.warning(f"Feature importance unavailable: {imp['error']}")
    else:
        st.caption(f"Method: *{imp.get('method', 'N/A')}*")
        names = imp.get("feature_names", [])
        values = imp.get("importances", imp.get("importances_mean", []))
        if names and values:
            imp_df = pd.DataFrame({"Feature": names, "Importance": values}).head(15)
            imp_df = imp_df.sort_values("Importance")
            fig = px.bar(
                imp_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig = apply_theme(fig, "Feature Importance — Top Features")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Permutation importance ─────────────────────────────────────────────────
    if df is not None:
        with st.expander("🔄 Permutation Importance (Model-Agnostic)"):
            st.caption(
                "Permutation importance shuffles each feature and measures the drop in PR-AUC. "
                "This is model-agnostic and works for any estimator. "
                "⚠️ May take 30–60 seconds for large datasets."
            )
            if st.button("Compute Permutation Importance"):
                from src.train import prepare_data
                X, y = prepare_data(df)
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
                )
                with st.spinner("Computing permutation importance..."):
                    perm = get_permutation_importance(pipeline, X_test, y_test)
                if "error" in perm:
                    st.warning(perm["error"])
                else:
                    st.caption(f"Method: *{perm['method']}*")
                    perm_df = pd.DataFrame({
                        "Feature": perm["feature_names"],
                        "Mean Decrease in PR-AUC": perm["importances_mean"],
                        "Std": perm["importances_std"],
                    }).sort_values("Mean Decrease in PR-AUC", ascending=False).head(10)
                    fig = px.bar(
                        perm_df.sort_values("Mean Decrease in PR-AUC"),
                        x="Mean Decrease in PR-AUC", y="Feature",
                        orientation="h",
                        error_x="Std",
                        color="Mean Decrease in PR-AUC",
                        color_continuous_scale="Oranges",
                    )
                    fig = apply_theme(fig, "Permutation Importance")
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if df is not None:
        with st.expander("🎯 SHAP Explanations (Global)"):
            st.caption(
                "SHAP (SHapley Additive exPlanations) assigns each feature a contribution "
                "to the prediction. ⚠️ May take several minutes for Kernel SHAP."
            )
            if st.button("Compute SHAP Values"):
                from src.train import prepare_data
                X, _ = prepare_data(df)
                sample = X.sample(min(100, len(X)), random_state=42)
                with st.spinner("Computing SHAP values (this may take a moment)..."):
                    shap_result = get_shap_values(pipeline, sample)
                if "error" in shap_result:
                    st.warning(f"SHAP unavailable: {shap_result['error']}")
                else:
                    st.caption(f"Method: *{shap_result['method']}* | Samples: {shap_result['n_samples']}")
                    shap_vals = np.array(shap_result["shap_values"])
                    feat_names = shap_result["feature_names"]
                    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                    shap_df = pd.DataFrame({
                        "Feature": feat_names[:len(mean_abs_shap)],
                        "Mean |SHAP|": mean_abs_shap,
                    }).sort_values("Mean |SHAP|", ascending=False).head(15)
                    fig = px.bar(
                        shap_df.sort_values("Mean |SHAP|"),
                        x="Mean |SHAP|", y="Feature",
                        orientation="h",
                        color="Mean |SHAP|",
                        color_continuous_scale="Purples",
                    )
                    fig = apply_theme(fig, "Global SHAP Feature Importance")
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE G: About
# ─────────────────────────────────────────────────────────────────────────────

def page_about():
    st.markdown("""
    <div class='page-header'>
        <h1>ℹ️ About the Project</h1>
        <p>Academic mini-project background, methodology, and technology stack.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Problem Statement

    Industrial machinery failures cause unplanned downtime, safety risks, and significant
    financial losses. **Predictive maintenance** uses sensor data and machine learning to
    anticipate failures before they occur — enabling timely intervention and reducing costs.

    ## 📊 Dataset

    This project uses the **UCI AI4I 2020 Predictive Maintenance Dataset**.

    > **⚠️ Important:** This dataset is **synthetic** — it was mathematically generated to
    > resemble realistic industrial conditions but does not originate from real machine telemetry.
    > It is intended for research and educational purposes.

    | Attribute | Detail |
    |---|---|
    | Records | 10,000 |
    | Features | 14 original columns |
    | Target | Machine failure (binary) |
    | Failure rate | ~3.4% |
    | Source | UCI ML Repository — ID 601 |

    **Citation:** Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

    ## 🧬 ML Workflow

    ```
    Raw CSV → Validation → Feature Engineering → Stratified Split (80/20)
    → Model Training (with class weights) → Evaluation (PR-AUC priority)
    → Saved Pipeline → Streamlit Dashboard / Flask API
    ```

    ## 🤖 Models Trained

    | Model | Notes |
    |---|---|
    | Logistic Regression | Baseline linear model, class_weight=balanced |
    | Random Forest | Ensemble of decision trees, class_weight=balanced |
    | SVM | RBF kernel, calibrated for probabilities |
    | Gradient Boosting | Boosted trees, subsample=0.8 |

    ## 🛠️ Technology Stack

    | Category | Technology |
    |---|---|
    | Language | Python 3.11 |
    | ML | scikit-learn |
    | Dashboard | Streamlit + Plotly |
    | REST API | Flask |
    | Explainability | SHAP, permutation importance |
    | Data | pandas, numpy |
    | Serialisation | joblib |

    ## 📐 Feature Engineering

    Three physics-informed derived features are added:
    - **Temperature difference [K]** = Process temp − Air temp
    - **Mechanical power [W]** = Torque × RPM × 2π/60
    - **Wear-torque interaction** = Tool wear × Torque

    These are derived inside a scikit-learn `Pipeline`, ensuring identical
    transformations at training time and inference time.

    ## ⚠️ Limitations

    - The model is trained on **synthetic** data, not real machine data.
    - Performance on real industrial equipment is **unknown** without validation.
    - A "No Failure" prediction does **not** guarantee machine safety.
    - Class imbalance (~3.4% failure rate) makes evaluation challenging.
    - Deployment in safety-critical environments requires certified validation.

    ## 🚀 Future Enhancements

    - Real-time telemetry ingestion (MQTT / Kafka)
    - Time-series failure prediction (LSTM / Transformer)
    - Drift detection and model retraining
    - Multi-class failure mode prediction
    - Federated learning for privacy-preserving multi-site deployment
    - Integration with CMMS (Computerised Maintenance Management Systems)

    ---
    *Built as an academic mini-project. Not for production safety-critical use.*
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main app entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    page, selected_model, threshold = render_sidebar()

    # Load data (cached)
    df, report, data_error = cached_load_data()
    registry = cached_load_registry()

    # Route pages
    if page == "🏠 Executive Overview":
        page_overview(df, report, registry, selected_model, threshold)
    elif page == "📊 Exploratory Data Analysis":
        page_eda(df)
    elif page == "🔮 Predict Machine Failure":
        page_predict(selected_model, threshold)
    elif page == "📈 Model Evaluation":
        page_evaluation()
    elif page == "📦 Batch Prediction":
        page_batch(selected_model, threshold)
    elif page == "🧠 Model Explainability":
        page_explainability(df, selected_model)
    elif page == "ℹ️ About the Project":
        page_about()


if __name__ == "__main__":
    main()
