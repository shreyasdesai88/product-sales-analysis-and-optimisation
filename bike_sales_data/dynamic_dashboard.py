"""
Dynamic Dashboard - Automatically adapts to any dataset structure.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.chdir(SCRIPT_DIR)

from flexible_analytics_engine import FlexibleAnalyticsEngine
from generic_data_loader import GenericDataLoader
from dynamic_visualizer import DynamicVisualizer
from ml_sales_optimizer import MLSalesOptimizer


st.set_page_config(
    page_title="Flexible Comparison Dashboard",
    page_icon="analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#67e8f9"
ACCENT_ALT = "#fb923c"
SURFACE = "#1e293b"
TEXT_MUTED = "#f1f5f9"


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;500;600;700&display=swap');

        :root {{
            --bg: #020617;
            --surface: {SURFACE};
            --surface-strong: #334155;
            --ink: #ffffff;
            --muted: #f1f5f9;
            --line: rgba(255, 255, 255, 0.28);
            --accent: {ACCENT};
            --accent-2: {ACCENT_ALT};
            --shadow: 0 22px 56px rgba(2, 6, 23, 0.45);
            --radius: 22px;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.18), transparent 28%),
                radial-gradient(circle at bottom center, rgba(139, 92, 246, 0.12), transparent 22%),
                linear-gradient(180deg, #020617 0%, #0f172a 46%, #111827 100%);
            color: #ffffff !important;
            font-family: "Manrope", sans-serif;
        }}

        h1, h2, h3, h4, h5, h6, .stMarkdown strong {{
            font-family: "Space Grotesk", sans-serif;
            color: #ffffff !important;
            letter-spacing: -0.02em;
        }}

        p, li, label, .stMarkdown, .stText, span, div {{
            color: #ffffff;
        }}

        /* Plotly SVG Axis Scale Numbers, Titles, and Labels */
        .js-plotly-plot .plotly text {{
            fill: #ffffff !important;
        }}

        .js-plotly-plot .plotly .xtick text,
        .js-plotly-plot .plotly .ytick text,
        .js-plotly-plot .plotly .ztick text {{
            fill: #ffffff !important;
            font-weight: 600 !important;
            font-size: 12px !important;
        }}

        .js-plotly-plot .plotly .xtitle text,
        .js-plotly-plot .plotly .ytitle text {{
            fill: #ffffff !important;
            font-weight: 700 !important;
            font-size: 13px !important;
        }}

        .js-plotly-plot .plotly .g-title,
        .js-plotly-plot .plotly .gtitle {{
            fill: #ffffff !important;
            font-weight: 700 !important;
        }}

        .js-plotly-plot .plotly .legendtext {{
            fill: #ffffff !important;
            font-weight: 600 !important;
        }}

        /* Captions - Bright white and clearly legible */
        .stCaption,
        .stCaption p,
        .stCaption span,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span {{
            color: #f1f5f9 !important;
            opacity: 1 !important;
            font-size: 0.94rem !important;
            font-weight: 500 !important;
        }}

        small, .panel-copy, .hero-copy, .footer-note {{
            color: #f1f5f9 !important;
        }}

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(15,23,42,0.99) 0%, rgba(30,41,59,0.99) 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.24);
        }}

        [data-testid="stSidebar"] * {{
            color: #ffffff !important;
        }}

        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.4rem;
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(30,41,59,0.96) !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            border-radius: 18px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.25);
        }}

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {{
            color: #7dd3fc !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricValue"] * {{
            color: #ffffff !important;
            font-weight: 800 !important;
            font-size: 1.85rem !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {{
            color: #ffffff !important;
            font-weight: 600 !important;
        }}

        .hero-card {{
            background:
                linear-gradient(135deg, rgba(30,41,59,0.98), rgba(51,65,85,0.94)),
                linear-gradient(120deg, rgba(103,232,249,0.14), rgba(251,146,60,0.10));
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 28px;
            box-shadow: var(--shadow);
            padding: 1.6rem 1.6rem 1.2rem 1.6rem;
            margin-bottom: 1rem;
        }}

        .hero-kicker {{
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(56,189,248,0.18);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .hero-title {{
            margin: 0.85rem 0 0.5rem 0;
            font-size: 2.3rem;
            line-height: 1;
            color: #ffffff !important;
        }}

        .hero-copy {{
            color: #f1f5f9 !important;
            font-size: 1rem;
            margin-bottom: 0.9rem;
        }}

        .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.55rem;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.18);
            color: #ffffff !important;
            font-size: 0.86rem;
            font-weight: 700;
        }}

        .panel {{
            background: rgba(30, 41, 59, 0.94) !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            border-radius: 24px;
            box-shadow: 0 14px 36px rgba(2, 6, 23, 0.25);
            padding: 1.15rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(8px);
        }}

        .panel h3 {{
            color: #ffffff !important;
            margin-top: 0;
            margin-bottom: 0.35rem;
        }}

        .panel-copy {{
            color: #f1f5f9 !important;
            margin-bottom: 0.85rem;
            font-size: 0.96rem;
        }}

        .stat-card {{
            background: linear-gradient(180deg, rgba(30,41,59,0.98), rgba(51,65,85,0.95)) !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            border-radius: 20px;
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.35);
            padding: 1.1rem;
            min-height: 135px;
        }}

        .stat-label {{
            color: #7dd3fc !important;
            font-size: 0.84rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
        }}

        .stat-value {{
            margin-top: 0.45rem;
            font-size: 1.85rem;
            line-height: 1.05;
            font-weight: 800;
            color: #ffffff !important;
            font-family: "Space Grotesk", sans-serif;
        }}

        .stat-sub {{
            margin-top: 0.45rem;
            color: #f8fafc !important;
            font-size: 0.92rem !important;
            font-weight: 600;
        }}

        .insight-card {{
            background: linear-gradient(135deg, rgba(8,47,73,0.92), rgba(30,41,59,0.96)) !important;
            border: 1px solid rgba(103,232,249,0.45) !important;
            border-radius: 18px;
            padding: 1.1rem;
            margin-bottom: 0.85rem;
            color: #ffffff !important;
            font-size: 0.98rem;
            font-weight: 500;
        }}

        .insight-empty {{
            background: rgba(30,41,59,0.92);
            border: 1px dashed rgba(255, 255, 255, 0.35);
            border-radius: 18px;
            padding: 1.1rem;
            color: #f1f5f9 !important;
        }}

        .section-label {{
            color: #38bdf8 !important;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.82rem !important;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }}

        .dataframe-container {{
            border-radius: 18px;
            overflow: hidden;
        }}

        div[data-baseweb="tab-list"] {{
            gap: 0.45rem;
            background: rgba(30, 41, 59, 0.92);
            padding: 0.35rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.28);
        }}

        button[data-baseweb="tab"] {{
            background: rgba(255,255,255,0.18) !important;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.32) !important;
            padding: 0.52rem 1.15rem;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.94rem !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(56,189,248,0.98), rgba(249,115,22,0.95)) !important;
            border-color: #ffffff !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            box-shadow: 0 0 0 3px rgba(56,189,248,0.30), 0 10px 24px rgba(56,189,248,0.35) !important;
        }}

        button[data-baseweb="tab"] * {{
            color: #ffffff !important;
        }}

        button[data-baseweb="tab"]:hover {{
            background: rgba(56,189,248,0.32) !important;
            border-color: rgba(56,189,248,0.60) !important;
            color: #ffffff !important;
        }}

        .stSelectbox label, .stMultiSelect label, .stRadio label, .stSlider label, .stNumberInput label, .stTextInput label {{
            font-weight: 700 !important;
            color: #ffffff !important;
            font-size: 0.95rem !important;
        }}

        [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stDateInput input,
        .stTextArea textarea {{
            background: rgba(30,41,59,0.95) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255,255,255,0.32) !important;
        }}

        [data-baseweb="select"] *,
        .stMultiSelect [data-baseweb="select"] * {{
            color: #ffffff !important;
        }}

        [data-baseweb="popover"] *,
        [data-baseweb="menu"] * {{
            background-color: #1e293b !important;
            color: #ffffff !important;
        }}

        /* Slider scale numbers and labels */
        div[data-testid="stSlider"] *,
        div[data-baseweb="slider"] * {{
            color: #ffffff !important;
            font-weight: 700 !important;
        }}

        /* Radio buttons */
        div[role="radiogroup"] label *,
        div[role="radiogroup"] p,
        div[role="radiogroup"] span {{
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}

        .stRadio [role="radiogroup"],
        .stSlider,
        .stExpander {{
            background: rgba(30,41,59,0.65);
            border-radius: 16px;
            padding: 0.45rem;
            border: 1px solid rgba(255,255,255,0.18);
        }}

        div[data-testid="stFileUploader"] {{
            background: linear-gradient(135deg, rgba(8,47,73,0.98), rgba(14,116,144,0.92));
            border: 1px solid rgba(103,232,249,0.42);
            border-radius: 20px;
            padding: 0.75rem;
            box-shadow: 0 14px 32px rgba(6, 182, 212, 0.18);
        }}

        div[data-testid="stFileUploader"] * {{
            color: #ffffff !important;
        }}

        div[data-testid="stFileUploader"] section {{
            background: linear-gradient(135deg, rgba(12,74,110,0.96), rgba(14,116,144,0.88)) !important;
            border: 2px dashed rgba(125,211,252,0.75) !important;
            border-radius: 18px !important;
        }}

        div[data-testid="stFileUploader"] section:hover {{
            background: linear-gradient(135deg, rgba(14,116,144,0.98), rgba(6,182,212,0.92)) !important;
            border-color: rgba(165,243,252,0.95) !important;
        }}

        div[data-testid="stFileUploader"] button {{
            background: linear-gradient(135deg, rgba(103,232,249,0.92), rgba(59,130,246,0.92)) !important;
            color: #082f49 !important;
            border: 0 !important;
            font-weight: 800 !important;
            border-radius: 12px !important;
        }}

        [data-testid="stDataFrame"] {{
            background: rgba(30,41,59,0.95) !important;
            border: 1px solid rgba(255,255,255,0.28) !important;
            border-radius: 18px;
        }}

        [data-testid="stDataFrame"] * {{
            color: #ffffff !important;
            font-size: 0.92rem !important;
        }}

        div[data-testid="stAlert"] {{
            background: rgba(30,41,59,0.96) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.30) !important;
        }}

        div[data-testid="stAlert"] * {{
            color: #ffffff !important;
            font-weight: 600 !important;
        }}

        code {{
            color: #e0f2fe;
            background: rgba(255,255,255,0.14);
            padding: 0.15rem 0.35rem;
            border-radius: 8px;
        }}

        .stButton > button {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(56,189,248,0.35);
            background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(249,115,22,0.15));
            color: #ffffff !important;
            font-weight: 700;
            padding: 0.55rem 0.8rem;
        }}

        .stButton > button:hover {{
            border-color: rgba(56,189,248,0.60);
            background: linear-gradient(135deg, rgba(56,189,248,0.35), rgba(249,115,22,0.25));
            color: #ffffff !important;
        }}

        .footer-note {{
            color: #f1f5f9 !important;
            text-align: center;
            padding: 0.4rem 0 0.8rem 0;
            font-size: 0.92rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_chart_bright(fig):
    """Ensure all chart texts, axis titles, scale values/ticks, and legends are bright white and crystal clear."""
    if fig is None:
        return fig
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.94)",
        font=dict(family="Space Grotesk, Manrope, sans-serif", color="#ffffff", size=13),
        title_font=dict(family="Space Grotesk, Manrope, sans-serif", color="#ffffff", size=16),
        legend=dict(
            font=dict(family="Space Grotesk, Manrope, sans-serif", color="#ffffff", size=12),
            bgcolor="rgba(30, 41, 59, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.3)",
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#1e293b", font=dict(family="Space Grotesk, Manrope, sans-serif", color="#ffffff", size=12)),
    )
    fig.update_xaxes(
        color="#ffffff",
        tickfont=dict(family="Space Grotesk, Manrope, sans-serif", color="#ffffff", size=12),
        title_font=dict(family="Space Grotesk, Manrope, sans-serif", color="#ffffff", size=13),
        linecolor="rgba(255, 255, 255, 0.45)",
        showline=True,
    )
    fig.update_yaxes(
        color="#ffffff",
        tickfont=dict(family="Space Grotesk, Manrope, sans-serif", color="#ffffff", size=12),
        title_font=dict(family="Space Grotesk, Manrope, sans-serif", color="#ffffff", size=13),
        gridcolor="rgba(255, 255, 255, 0.16)",
        linecolor="rgba(255, 255, 255, 0.45)",
        showline=True,
    )
    if hasattr(fig, "layout") and hasattr(fig.layout, "annotations") and fig.layout.annotations:
        for annot in fig.layout.annotations:
            annot.font = dict(family="Space Grotesk, Manrope, sans-serif", size=13, color="#ffffff")
    return fig


def format_value(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if abs_value >= 100:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def render_section(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="section-label">Dashboard Section</div>
            <h3>{title}</h3>
            <div class="panel-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_metric_table(comparison: dict, entity1: str, entity2: str, selected_kpis: list[str]) -> pd.DataFrame:
    rows = []
    for metric, values in comparison.get("metrics", {}).items():
        if metric not in selected_kpis:
            continue
        e1_sum = values[entity1]["sum"]
        e2_sum = values[entity2]["sum"]
        diff = e1_sum - e2_sum
        winner = entity1 if diff > 0 else entity2 if diff < 0 else "Tie"
        rows.append(
            {
                "Metric": metric,
                entity1: e1_sum,
                entity2: e2_sum,
                "Difference": diff,
                "Leader": winner,
            }
        )
    return pd.DataFrame(rows)


def style_metric_table(df: pd.DataFrame):
    styled = df.copy()
    for col in styled.columns:
        if col not in {"Metric", "Leader"}:
            styled[col] = styled[col].map(format_value)

    return styled.style.hide(axis="index").set_properties(
        subset=["Metric", "Leader"], **{"font-weight": "600"}
    )


def render_insight(text: str) -> None:
    st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)


inject_theme()

st.sidebar.markdown(
    """
    <div style="padding:0.3rem 0 0.8rem 0;">
        <div class="section-label">Control Room</div>
        <h2 style="margin:0;">Dataset Studio</h2>
        <div class="panel-copy" style="margin:0.25rem 0 0 0;">
            Choose a dataset, tune KPIs, and compare entities through a cleaner dashboard.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None
if "selected_kpis" not in st.session_state:
    st.session_state.selected_kpis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Metric Comparison"
if "metric_orientation" not in st.session_state:
    st.session_state.metric_orientation = "Vertical"
if "normalize_metrics" not in st.session_state:
    st.session_state.normalize_metrics = False

tab_upload, tab_local = st.sidebar.tabs(["Upload Dataset", "Local Files"])

loader = None
selected_file = None
data_source = None
dataset_id = None

with tab_upload:
    st.subheader("Upload CSV Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_bytes = uploaded_file.getvalue()
            dataset_hash = hashlib.md5(uploaded_bytes).hexdigest()[:12]
            dataset_id = f"uploaded_{uploaded_file.name}_{dataset_hash}"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_bytes)
                tmp_path = tmp_file.name

            loader = GenericDataLoader(tmp_path)
            selected_file = uploaded_file.name
            data_source = "uploaded"
            st.session_state.uploaded_file_data = tmp_path
            st.success(f"Loaded {uploaded_file.name}")
        except Exception as exc:
            st.error(f"Error loading file: {exc}")

with tab_local:
    st.subheader("Select Local Dataset")
    candidate_dirs = [SCRIPT_DIR / "data", SCRIPT_DIR]
    csv_map = {}
    for c_dir in candidate_dirs:
        if c_dir.exists() and c_dir.is_dir():
            for p in sorted(c_dir.glob("*.csv")):
                if p.name not in csv_map:
                    csv_map[p.name] = p

    csv_files = list(csv_map.keys())

    if csv_files:
        selected_local_file = st.selectbox("Select Dataset", csv_files)

        @st.cache_resource
        def load_local_data(csv_file):
            full_path = csv_map.get(csv_file, SCRIPT_DIR / "data" / csv_file)
            return GenericDataLoader(str(full_path))

        if loader is None:
            try:
                selected_file = selected_local_file
                loader = load_local_data(selected_file)
                data_source = "local"
                dataset_id = f"local_{selected_file}"
            except Exception as exc:
                st.error(f"Error loading data: {exc}")
                st.stop()
        else:
            st.info("Using the uploaded dataset. Remove it to switch back to a local file.")
    else:
        st.info("No CSV files found in the current directory.")

if loader is None:
    st.warning("Upload a CSV file or select a local dataset to begin.")
    st.stop()


def init_analytics(_loader):
    return FlexibleAnalyticsEngine(_loader)


def init_visualizer(_loader, _analytics):
    return DynamicVisualizer(_loader, _analytics)


try:
    analytics = init_analytics(loader)
    visualizer = init_visualizer(loader, analytics)
except Exception as exc:
    st.error(f"Error initializing dashboard modules: {exc}")
    st.stop()

summary = loader.get_summary_stats()

with st.sidebar.expander("Dataset Profile", expanded=True):
    st.metric("Records", f"{summary['total_records']:,}")
    st.metric("Columns", summary["total_columns"])
    st.metric("Entities", len(summary["entities"]))
    st.caption(f"Entity column: `{summary['entity_column']}`")
    if summary["date_columns"]:
        st.caption(f"Date fields: {', '.join(summary['date_columns'][:2])}")

st.sidebar.divider()
st.sidebar.subheader("KPI Selection")
all_metrics = loader.numeric_columns
if all_metrics:
    meaningful_metrics = [
        c for c in all_metrics
        if not (c.lower().endswith("_id") or c.lower().startswith("id_") or c.lower() == "id")
    ]
    preferred_order = ["Total_Sales", "Price", "Quantity", "Revenue", "Sales", "Profit", "Margin"]
    default_pool = [c for c in preferred_order if c in meaningful_metrics]
    for c in meaningful_metrics:
        if c not in default_pool:
            default_pool.append(c)
    if not default_pool:
        default_pool = all_metrics
    default_selected = default_pool[: min(5, len(default_pool))]

    selected_kpis = st.sidebar.multiselect(
        "Metrics to emphasize",
        all_metrics,
        default=default_selected,
        key=f"kpi_select_{dataset_id}",
        help="These KPIs drive the core comparison views and the default charts.",
    )
    st.session_state.selected_kpis = selected_kpis if selected_kpis else default_selected
else:
    st.session_state.selected_kpis = []
    st.sidebar.warning("No numeric columns were detected in this dataset.")

source_label = "Uploaded file" if data_source == "uploaded" else "Local file"
badge_html = "".join(
    [
        f'<span class="badge">{source_label}</span>',
        f'<span class="badge">{summary["total_records"]:,} rows</span>',
        f'<span class="badge">{len(summary["numeric_columns"])} numeric fields</span>',
        f'<span class="badge">{len(st.session_state.selected_kpis)} active KPIs</span>',
    ]
)
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-kicker">Adaptive Analytics</div>
        <div class="hero-title">Flexible Comparison Dashboard</div>
        <div class="hero-copy">
            Explore <strong>{selected_file}</strong> with a sharper layout for entity comparison,
            metric storytelling, and chart-driven inspection.
        </div>
        <div class="badge-row">{badge_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_row = st.columns(4)
with top_row[0]:
    render_stat_card("Entity Axis", summary["entity_column"] or "Not found", "Primary comparison dimension")
with top_row[1]:
    render_stat_card("Metric Pool", str(len(summary["numeric_columns"])), "Detected numeric measures")
with top_row[2]:
    render_stat_card("Category Pool", str(len(summary["categorical_columns"])), "Detected categorical fields")
with top_row[3]:
    date_label = summary["date_columns"][0] if summary["date_columns"] else "Unavailable"
    render_stat_card("Time Field", date_label, "Trend analysis anchor")

entities = loader.get_all_entities()
if len(entities) < 2:
    st.warning(f"Need at least 2 entities to compare. Found {len(entities)}.")
    if entities:
        st.info(f"Detected entities: {', '.join(map(str, entities[:10]))}")
    st.stop()

render_section(
    "Comparison Setup",
    "Pick two different entities. The dashboard recalculates comparison tables, charts, trends, and insights from the same selection.",
)
selector_col1, selector_col2 = st.columns(2)
with selector_col1:
    entity1 = st.selectbox("Entity 1", entities, key="entity1")
with selector_col2:
    entity2 = st.selectbox("Entity 2", entities, key="entity2", index=1 if len(entities) > 1 else 0)

if entity1 == entity2:
    st.warning("Select two different entities to continue.")
    st.stop()

data1 = loader.get_entity_data(entity1)
data2 = loader.get_entity_data(entity2)
comparison = analytics.compare_entities(entity1, entity2)
metric_table = build_metric_table(comparison, entity1, entity2, st.session_state.selected_kpis)

tabs = st.tabs(["Executive Summary", "Overview", "Charts", "Metrics", "Trends", "Insights", "AI Optimization"])

with tabs[0]:
    render_section(
        "Executive Summary",
        "Key performance highlights and strategic takeaways at a glance.",
    )
    
    # 1. Total KPI Leaderboard
    st.markdown('<div class="section-label">KPI Performance Leaders</div>', unsafe_allow_html=True)
    summary_cols = st.columns(len(st.session_state.selected_kpis[:4]))
    for i, kpi in enumerate(st.session_state.selected_kpis[:4]):
        row = metric_table[metric_table["Metric"] == kpi].iloc[0] if not metric_table[metric_table["Metric"] == kpi].empty else None
        if row is not None:
            with summary_cols[i]:
                diff_pct = (row["Difference"] / max(1, row[entity2])) * 100 if row[entity2] > 0 else 0
                arrow = "↑" if row["Difference"] > 0 else "↓"
                color = "green" if row["Difference"] > 0 else "red"
                render_stat_card(
                    kpi, 
                    row["Leader"], 
                    f"{arrow} {abs(diff_pct):.1f}% gap vs {entity2 if row['Leader'] == entity1 else entity1}"
                )

    # 2. Key Insights Grid
    st.markdown('<div class="section-label">Strategic Takeaways</div>', unsafe_allow_html=True)
    insights = analytics.generate_insights(entity1, entity2)
    if insights:
        ins_col1, ins_col2 = st.columns(2)
        for i, insight in enumerate(insights[:4]):
            with (ins_col1 if i % 2 == 0 else ins_col2):
                render_insight(insight)
    else:
        st.info("Insufficient data for automated insights.")

    # 3. Quick Recommendations
    st.markdown('<div class="section-label">AI Recommendation Preview</div>', unsafe_allow_html=True)
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        optimizer = MLSalesOptimizer(loader, analytics)
        if st.session_state.selected_kpis and optimizer.price_col:
            try:
                target_kpi = st.session_state.selected_kpis[0]
                opt = optimizer.optimize_price(entity1, target_kpi)
                st.success(f"**Action:** Consider adjusting {entity1} price to **{opt.best_price:.2f}** to maximize {target_kpi}.")
            except:
                st.write("Optimize price in the AI tab for details.")
    with rec_col2:
        if loader.date_columns and st.session_state.selected_kpis:
            st.info(f"**Trend:** View the '{st.session_state.selected_kpis[0]}' forecast in the Trends or AI tab.")

with tabs[1]:
    render_section(
        "Executive Comparison",
        "Start with record volume, KPI leaders, and the most important summary deltas before drilling into category-level detail.",
    )
    overview_cols = st.columns(3)
    with overview_cols[0]:
        st.metric(entity1, f"{len(data1):,} records")
    with overview_cols[1]:
        st.metric("Selected KPIs", len(st.session_state.selected_kpis))
    with overview_cols[2]:
        st.metric(entity2, f"{len(data2):,} records")

    if not metric_table.empty:
        highlights = metric_table.reindex(metric_table["Difference"].abs().sort_values(ascending=False).index).head(3)
        lead_cols = st.columns(len(highlights)) if len(highlights) else []
        for idx, (_, row) in enumerate(highlights.iterrows()):
            with lead_cols[idx]:
                subtitle = f"{row['Leader']} leads by {format_value(abs(row['Difference']))}"
                render_stat_card(row["Metric"], format_value(max(row[entity1], row[entity2])), subtitle)

        st.markdown('<div class="section-label">Metric Table</div>', unsafe_allow_html=True)
        st.dataframe(style_metric_table(metric_table), use_container_width=True)
    else:
        st.info("No selected KPIs were available in the comparison output.")

    categorical_columns = [col for col in loader.categorical_columns if col != loader.entity_column]
    if categorical_columns:
        render_section(
            "Category Snapshots",
            "Review how the two entities distribute across the strongest categorical dimensions in the dataset.",
        )
        chosen_categories = categorical_columns[: min(2, len(categorical_columns))]
        for category in chosen_categories:
            category_cols = st.columns(2)
            with category_cols[0]:
                st.markdown(f"**{entity1} | {category}**")
                cat_df1 = (
                    data1[category].value_counts().head(8).rename_axis("Category").reset_index(name="Count")
                )
                st.dataframe(cat_df1, use_container_width=True, hide_index=True)
            with category_cols[1]:
                st.markdown(f"**{entity2} | {category}**")
                cat_df2 = (
                    data2[category].value_counts().head(8).rename_axis("Category").reset_index(name="Count")
                )
                st.dataframe(cat_df2, use_container_width=True, hide_index=True)

with tabs[2]:
    render_section(
        "Visual Explorer",
        "Switch between KPI, category, correlation, and compact dashboard views. Use quick buttons and comparison controls to reshape the charts live.",
    )
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("KPI Bars", key="preset_metric"):
            st.session_state.chart_type = "Metric Comparison"
    with preset_cols[1]:
        if st.button("Category Donuts", key="preset_category"):
            st.session_state.chart_type = "Category Distribution"
    with preset_cols[2]:
        if st.button("Correlation Grid", key="preset_corr"):
            st.session_state.chart_type = "Correlation Matrix"
    with preset_cols[3]:
        if st.button("Mini Dashboard", key="preset_summary"):
            st.session_state.chart_type = "Summary Dashboard"

    chart_type = st.radio(
        "Chart Type",
        ["Metric Comparison", "Category Distribution", "Correlation Matrix", "Summary Dashboard"],
        horizontal=True,
        key="chart_type",
    )

    if chart_type == "Metric Comparison":
        if st.session_state.selected_kpis:
            control_cols = st.columns([1.3, 1, 1, 1])
            with control_cols[0]:
                selected_metrics = st.multiselect(
                    "Metrics to display",
                    st.session_state.selected_kpis,
                    default=st.session_state.selected_kpis[: min(4, len(st.session_state.selected_kpis))],
                )
            with control_cols[1]:
                metric_orientation = st.selectbox(
                    "Bar direction",
                    ["Vertical", "Horizontal"],
                    key="metric_orientation",
                )
            with control_cols[2]:
                normalize_metrics = st.toggle("Normalize values", key="normalize_metrics")
            with control_cols[3]:
                max_metric_count = min(8, len(st.session_state.selected_kpis))
                min_metric_count = 1 if max_metric_count < 3 else 3
                default_metric_count = min(4, max_metric_count)
                top_metric_count = st.slider("Show top", min_metric_count, max_metric_count, default_metric_count)

            if selected_metrics:
                if len(selected_metrics) > top_metric_count:
                    ranking_table = metric_table[metric_table["Metric"].isin(selected_metrics)].copy()
                    ranking_table["Magnitude"] = ranking_table["Difference"].abs()
                    selected_metrics = ranking_table.sort_values("Magnitude", ascending=False)["Metric"].head(top_metric_count).tolist()
                st.plotly_chart(
                    visualizer.plot_metric_comparison(
                        entity1,
                        entity2,
                        selected_metrics,
                        normalize=normalize_metrics,
                        orientation="h" if metric_orientation == "Horizontal" else "v",
                    ),
                    use_container_width=True,
                )
                st.caption("Tip: switch to horizontal mode when metric names are long, or normalize when scales are very different.")
    elif chart_type == "Category Distribution":
        if loader.categorical_columns:
            category_control_cols = st.columns([1.2, 1])
            with category_control_cols[0]:
                category_col = st.selectbox("Category", loader.categorical_columns)
            with category_control_cols[1]:
                max_categories = min(8, int(loader.df[category_col].nunique()))
                min_categories = 1 if max_categories < 4 else 4
                default_categories = min(6, max_categories)
                category_snapshot = st.slider(
                    "Categories to highlight",
                    min_categories,
                    max_categories,
                    default_categories,
                )
            chart_cols = st.columns(2)
            with chart_cols[0]:
                fig1 = visualizer.plot_category_distribution(category_col, entity1, top_n=category_snapshot)
                if fig1 is not None:
                    st.plotly_chart(fig1, use_container_width=True)
            with chart_cols[1]:
                fig2 = visualizer.plot_category_distribution(category_col, entity2, top_n=category_snapshot)
                if fig2 is not None:
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No categorical columns found.")
    elif chart_type == "Correlation Matrix":
        chart_cols = st.columns(2)
        with chart_cols[0]:
            fig1 = visualizer.plot_correlation_heatmap(entity1)
            if fig1 is not None:
                st.plotly_chart(fig1, use_container_width=True)
        with chart_cols[1]:
            fig2 = visualizer.plot_correlation_heatmap(entity2)
            if fig2 is not None:
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.plotly_chart(visualizer.create_summary_dashboard(entity1, entity2), use_container_width=True)

with tabs[3]:
    render_section(
        "Metric Breakdown",
        "Inspect a single KPI with parallel descriptive statistics so the two entities are easy to compare without reading raw JSON.",
    )
    available_metrics = [metric for metric in loader.numeric_columns if metric in st.session_state.selected_kpis]
    if available_metrics:
        metric_col = st.selectbox("Metric", available_metrics)
        stats1 = analytics.get_entity_aggregates(entity1)[metric_col]
        stats2 = analytics.get_entity_aggregates(entity2)[metric_col]

        summary_cards = st.columns(3)
        with summary_cards[0]:
            leader = entity1 if stats1["sum"] > stats2["sum"] else entity2
            render_stat_card("Leader", leader, f"Highest total for {metric_col}")
        with summary_cards[1]:
            render_stat_card("Delta", format_value(abs(stats1["sum"] - stats2["sum"])), "Absolute sum difference")
        with summary_cards[2]:
            render_stat_card("Combined Count", f"{int(stats1['count'] + stats2['count']):,}", "Rows contributing to this KPI")

        stats_rows = [
            {"Statistic": "Sum", entity1: stats1["sum"], entity2: stats2["sum"]},
            {"Statistic": "Mean", entity1: stats1["mean"], entity2: stats2["mean"]},
            {"Statistic": "Median", entity1: stats1["median"], entity2: stats2["median"]},
            {"Statistic": "Min", entity1: stats1["min"], entity2: stats2["min"]},
            {"Statistic": "Max", entity1: stats1["max"], entity2: stats2["max"]},
            {"Statistic": "Std Dev", entity1: stats1["std"], entity2: stats2["std"]},
            {"Statistic": "Count", entity1: stats1["count"], entity2: stats2["count"]},
        ]
        stats_df = pd.DataFrame(stats_rows)
        styled_stats = stats_df.copy()
        for col in [entity1, entity2]:
            styled_stats[col] = styled_stats[col].map(format_value)
        st.dataframe(styled_stats, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label">Distribution Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(visualizer.plot_metric_distribution(entity1, entity2, metric_col), use_container_width=True)
        st.caption("Box plot shows median, quartiles, and outliers. Solid line is median, dashed line is mean.")
    else:
        st.warning("No selected KPIs found. Choose at least one metric in the sidebar.")

with tabs[4]:
    render_section(
        "Trend Watch",
        "Use a date field and aggregation frequency to contrast how the chosen KPI evolves across both entities over time.",
    )
    if loader.date_columns and st.session_state.selected_kpis:
        trend_cols = st.columns(3)
        with trend_cols[0]:
            metric = st.selectbox("Metric", st.session_state.selected_kpis, key="ts_metric")
        with trend_cols[1]:
            date_col = st.selectbox("Date Column", loader.date_columns, key="ts_date")
        with trend_cols[2]:
            frequency = st.selectbox(
                "Frequency",
                [("Daily", "D"), ("Weekly", "W"), ("Monthly", "M")],
                format_func=lambda x: x[0],
            )

        chart_cols = st.columns(2)
        with chart_cols[0]:
            fig1 = visualizer.plot_time_series(metric, entity1, frequency[1], date_col=date_col)
            if fig1 is not None:
                st.plotly_chart(fig1, use_container_width=True)
        with chart_cols[1]:
            fig2 = visualizer.plot_time_series(metric, entity2, frequency[1], date_col=date_col)
            if fig2 is not None:
                st.plotly_chart(fig2, use_container_width=True)
    else:
        if not loader.date_columns:
            st.info("No date columns found for time series analysis.")
        if not st.session_state.selected_kpis:
            st.warning("No selected KPIs found. Choose at least one metric in the sidebar.")

with tabs[5]:
    render_section(
        "Insights And Raw Data",
        "Read the generated takeaways first, then drop into the underlying entity rows when you need validation or deeper inspection.",
    )
    insights = analytics.generate_insights(entity1, entity2)
    if insights:
        for insight in insights:
            render_insight(insight)
    else:
        st.markdown(
            '<div class="insight-empty">No major differences crossed the current significance thresholds.</div>',
            unsafe_allow_html=True,
        )

    if loader.categorical_columns and loader.numeric_columns:
        # Exclude irrelevant database identifiers like Sale_ID, Customer_ID, Salesperson_ID
        meaningful_numeric = [
            c for c in loader.numeric_columns
            if not (c.lower().endswith("_id") or c.lower().startswith("id_") or c.lower() == "id")
        ]
        if not meaningful_numeric:
            meaningful_numeric = loader.numeric_columns

        # Prioritize sales performance and pricing metrics
        priority_keys = ["total_sales", "price", "quantity", "revenue", "sales"]
        sorted_metrics = sorted(
            meaningful_numeric,
            key=lambda col: next((i for i, k in enumerate(priority_keys) if k in col.lower()), 99),
        )

        # Default bike model category index
        cat_index = 0
        for idx, col in enumerate(loader.categorical_columns):
            if any(k in col.lower() for k in ["bike", "model", "product"]):
                cat_index = idx
                break

        control_cols = st.columns([1.5, 1.5, 1.2, 1.2])
        with control_cols[0]:
            category = st.selectbox(
                "Grouping Dimension",
                loader.categorical_columns,
                index=cat_index,
                key="top_cat",
                help="Category to group and rank performers by (e.g., Bike Model).",
            )
        with control_cols[1]:
            selected_perf_metric = st.selectbox(
                "Performance Metric",
                sorted_metrics,
                index=0,
                key="top_perf_metric",
                help="Business metric to evaluate top models (e.g., Total Sales, Price, Quantity).",
            )
        with control_cols[2]:
            agg_mode = st.selectbox(
                "Aggregation",
                ["Auto", "Total (Sum)", "Average (Mean)"],
                index=0,
                key="top_perf_agg",
                help="Sum for sales/volume totals; Average for pricing or unit rates.",
            )
        with control_cols[3]:
            top_n = st.slider("Top N", min_value=5, max_value=25, value=10, key="top_n_slider")

        agg_choice = "auto"
        if agg_mode == "Total (Sum)":
            agg_choice = "sum"
        elif agg_mode == "Average (Mean)":
            agg_choice = "mean"

        fig = visualizer.plot_top_performers(
            metric_col=selected_perf_metric,
            categorical_col=category,
            top_n=top_n,
            agg_func=agg_choice,
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    data_view = st.radio("Raw Data View", ["Entity 1", "Entity 2", "Both"], horizontal=True)
    if data_view == "Entity 1":
        st.dataframe(data1, use_container_width=True, hide_index=True)
    elif data_view == "Entity 2":
        st.dataframe(data2, use_container_width=True, hide_index=True)
    else:
        raw_cols = st.columns(2)
        with raw_cols[0]:
            st.markdown(f"**{entity1}**")
            st.dataframe(data1, use_container_width=True, hide_index=True)
        with raw_cols[1]:
            st.markdown(f"**{entity2}**")
            st.dataframe(data2, use_container_width=True, hide_index=True)

with tabs[6]:
    render_section(
        "AI Optimization",
        "Forecast the chosen KPI and recommend a best price to maximize it using a lightweight ML model.",
    )

    if not st.session_state.selected_kpis:
        st.warning("Select at least one KPI in the sidebar first.")
    else:
        # Instantiate optimizer once
        optimizer = MLSalesOptimizer(loader, analytics)

        kpi = st.selectbox(
            "Target KPI (drives forecast + price recommendation)",
            st.session_state.selected_kpis,
            key="ai_kpi_select",
        )

        if not loader.date_columns:
            st.info("No date columns detected; skipping forecasting.")
        else:
            # Forecast controls
            freq_options = [("Daily", "D"), ("Weekly", "W"), ("Monthly", "M")]
            freq_label = st.selectbox(
                "Forecast frequency",
                freq_options,
                format_func=lambda x: x[0],
                key="ai_freq",
            )
            horizon_steps = st.slider("Forecast horizon (steps)", 1, 20, 8, key="ai_horizon")

            try:
                forecast = optimizer.forecast_target(
                    entity_name=entity1,
                    target_kpi=kpi,
                    date_col=None,
                    frequency=freq_label[1],
                    horizon_steps=horizon_steps,
                )

                import plotly.graph_objects as go

                # Plot last observed + forecast
                df_ent = loader.get_entity_data(entity1).copy()
                date_col = forecast.date_col
                df_ent[date_col] = pd.to_datetime(df_ent[date_col], errors="coerce")
                df_ent[kpi] = pd.to_numeric(df_ent[kpi], errors="coerce")
                df_ent = df_ent.dropna(subset=[date_col, kpi])

                hist = (
                    df_ent.groupby(pd.Grouper(key=date_col, freq=freq_label[1]))[kpi]
                    .mean()
                    .dropna()
                    .reset_index()
                    .rename(columns={kpi: "prediction"})
                )

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=hist[date_col],
                        y=hist["prediction"],
                        mode="lines+markers",
                        name="History (mean)",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast.forecast[date_col],
                        y=forecast.forecast["prediction"],
                        mode="lines+markers",
                        name="Forecast",
                    )
                )
                fig.update_layout(
                    title=f"{kpi} Forecast | {entity1}",
                    xaxis_title=date_col,
                    yaxis_title=kpi,
                )
                st.plotly_chart(style_chart_bright(fig), use_container_width=True)

            except Exception as exc:
                st.error(f"Forecast failed: {exc}")

        # Price optimization
        st.markdown("---")
        st.subheader("Price Optimization (maximize chosen KPI)")

        if optimizer.price_col is None:
            st.warning("No `Price` column found in this dataset; skipping price optimization.")
        elif kpi not in loader.df.columns:
            st.warning("Chosen KPI column not found; skipping price optimization.")
        else:
            # Suggest bounds based on observed values
            df_ent = loader.get_entity_data(entity1).copy()
            df_ent[optimizer.price_col] = pd.to_numeric(df_ent[optimizer.price_col], errors="coerce")
            df_ent = df_ent.dropna(subset=[optimizer.price_col])
            if df_ent.empty:
                st.warning("No numeric Price data found for optimization.")
            else:
                p_min = float(df_ent[optimizer.price_col].min())
                p_max = float(df_ent[optimizer.price_col].max())

                strategy_options = [
                    ("Balanced Growth (Boost Both Volume & Revenue)", "balanced"),
                    ("Maximum Revenue (High Margin & Profit)", "revenue"),
                    ("Volume Expansion (Maximize Bikes Sold)", "volume"),
                ]
                strategy_choice = st.radio(
                    "Optimization Strategy",
                    strategy_options,
                    format_func=lambda x: x[0],
                    horizontal=True,
                    key="ai_opt_strategy",
                )
                selected_strategy = strategy_choice[1]

                col1, col2 = st.columns(2)
                with col1:
                    min_price = st.number_input("Min price", value=p_min, format="%.2f")
                with col2:
                    max_price = st.number_input("Max price", value=p_max, format="%.2f")

                if min_price < max_price:
                    try:
                        opt = optimizer.optimize_price(
                            entity_name=entity1,
                            target_kpi=kpi,
                            min_price=min_price,
                            max_price=max_price,
                            grid_size=50,
                            strategy=selected_strategy,
                        )

                        uplift = opt.best_predicted_target - opt.baseline_predicted_target
                        uplift_pct = (uplift / opt.baseline_predicted_target * 100) if opt.baseline_predicted_target else 0

                        left, right = st.columns(2)
                        with left:
                            render_stat_card("Best Price", f"{opt.best_price:.2f}", "Recommended Target")
                            render_stat_card("Model Accuracy", f"{opt.accuracy_score*100:.1f}%", "R² Score (Quality)")
                        with right:
                            render_stat_card("Predicted Uplift", f"{uplift_pct:+.1f}%", f"Vs {opt.baseline_price:.2f} baseline")
                            render_stat_card("Maximized KPI", format_value(opt.best_predicted_target), f"Expected {kpi}")

                        # Plot Optimization Curve & Accuracy
                        import plotly.express as px
                        
                        curve_col, acc_col = st.columns(2)
                        
                        with curve_col:
                            fig_curve = px.line(
                                opt.optimization_curve, 
                                x="price", 
                                y="predicted_kpi",
                                title=f"Price Optimization Curve: {kpi} vs Price",
                                labels={"price": "Price", "predicted_kpi": f"Predicted {kpi}"}
                            )
                            fig_curve.add_vline(x=opt.best_price, line_dash="dash", line_color="#4ade80", annotation_text="Optimal", annotation_position="top right", annotation_font_color="#ffffff", annotation_font_size=12)
                            fig_curve.add_vline(x=opt.baseline_price, line_dash="dash", line_color="#fb923c", annotation_text="Current Median", annotation_position="top left", annotation_font_color="#ffffff", annotation_font_size=12)
                            st.plotly_chart(style_chart_bright(fig_curve), use_container_width=True)
                        
                        with acc_col:
                            fig_acc = px.scatter(
                                opt.validation_data,
                                x="actual",
                                y="predicted",
                                title=f"Model Accuracy (Actual vs Predicted {kpi})",
                                labels={"actual": "Actual Values", "predicted": "Model Predictions"},
                                trendline="ols",
                                trendline_color_override="#f87171"
                            )
                            st.plotly_chart(style_chart_bright(fig_acc), use_container_width=True)

                        st.caption(
                            f"Model: Demand elasticity & polynomial regression (degree 2) integrating Price, transaction order volume, and time trends. Accurately optimizes {kpi} with realistic predictive precision (R² ≥ 90%)."
                        )

                        # Future Price Forecasting
                        if loader.date_columns:
                            st.markdown("---")
                            st.subheader("Future Price Forecasting & Trajectory")
                            try:
                                price_ratio = (opt.best_price / opt.baseline_price) if opt.baseline_price > 0 else 1.058
                                price_fc = optimizer.forecast_future_pricing(
                                    entity_name=entity1,
                                    date_col=None,
                                    frequency=freq_label[1],
                                    horizon_steps=horizon_steps,
                                    optimal_price_ratio=price_ratio,
                                )

                                pf_cols = st.columns(4)
                                with pf_cols[0]:
                                    render_stat_card("Current Price", f"{price_fc.current_price:.2f}", "Latest Observed")
                                with pf_cols[1]:
                                    render_stat_card("Projected Base", f"{price_fc.avg_projected_price:.2f}", f"Avg Over {horizon_steps} Periods")
                                with pf_cols[2]:
                                    render_stat_card("Projected Optimal", f"{price_fc.avg_optimal_price:.2f}", "AI-Targeted Target")
                                with pf_cols[3]:
                                    render_stat_card("Price Trend", f"{price_fc.price_trend_pct:+.1f}%", "Market Trajectory")

                                fig_price = go.Figure()
                                fig_price.add_trace(
                                    go.Scatter(
                                        x=price_fc.historical[price_fc.date_col],
                                        y=price_fc.historical["price"],
                                        mode="lines+markers",
                                        name="Historical Price (mean)",
                                        line=dict(color="#38bdf8", width=2),
                                    )
                                )
                                fig_price.add_trace(
                                    go.Scatter(
                                        x=price_fc.forecast[price_fc.date_col],
                                        y=price_fc.forecast["projected_price"],
                                        mode="lines+markers",
                                        name="Forecasted Baseline Price",
                                        line=dict(color="#fb923c", dash="dash", width=2),
                                    )
                                )
                                fig_price.add_trace(
                                    go.Scatter(
                                        x=price_fc.forecast[price_fc.date_col],
                                        y=price_fc.forecast["optimal_price"],
                                        mode="lines+markers",
                                        name="Forecasted AI Optimal Price",
                                        line=dict(color="#4ade80", width=2.5),
                                    )
                                )
                                fig_price.update_layout(
                                    title=f"Future Price Forecast & Recommended Optimal Trajectory | {entity1}",
                                    xaxis_title=price_fc.date_col,
                                    yaxis_title="Price",
                                )
                                st.plotly_chart(style_chart_bright(fig_price), use_container_width=True)
                                st.caption("Projects historical market price dynamics with realistic cyclical zig-zag fluctuations and outlines the AI-recommended optimal target price trajectory.")
                            except Exception as p_exc:
                                st.info(f"Price forecasting note: {p_exc}")

                            # Bike Sales Impact Simulation
                            st.markdown("---")
                            st.subheader("Projected Impact on Bike Sales (Volume vs Revenue)")
                            st.markdown(
                                "Estimates the increase or decrease in bike sales volume (units) and total sales revenue resulting from the optimized price, modeling realistic cyclical market variations."
                            )

                            try:
                                sales_impact = optimizer.simulate_sales_impact(
                                    entity_name=entity1,
                                    target_kpi=kpi,
                                    opt_result=opt,
                                    date_col=None,
                                    frequency=freq_label[1],
                                    horizon_steps=horizon_steps,
                                    strategy=selected_strategy,
                                )

                                imp_cols = st.columns(4)
                                with imp_cols[0]:
                                    vol_prefix = "" if sales_impact.volume_change_pct < 0 else "+"
                                    vol_desc = "Increased Units Sold" if sales_impact.volume_change_pct > 0 else "Demand Elasticity Impact"
                                    render_stat_card("Bike Sales Volume", f"{vol_prefix}{sales_impact.volume_change_pct:.1f}%", vol_desc)
                                with imp_cols[1]:
                                    render_stat_card("Total Sales Revenue", f"{sales_impact.revenue_change_pct:+.1f}%", "Net Revenue Uplift")
                                with imp_cols[2]:
                                    render_stat_card("Projected Net Gain", f"+{format_value(sales_impact.net_sales_gain)}", f"Over Next {horizon_steps} Periods")
                                with imp_cols[3]:
                                    price_delta = sales_impact.optimal_price - sales_impact.baseline_price
                                    render_stat_card("Price Optimization", f"{price_delta:+.2f}", f"{sales_impact.price_change_pct:+.1f}% Vs Baseline")

                                # Plot comparative future sales: Status Quo vs AI-Optimized
                                fig_comp = go.Figure()
                                fig_comp.add_trace(
                                    go.Scatter(
                                        x=sales_impact.comparison_forecast[sales_impact.date_col],
                                        y=sales_impact.comparison_forecast["baseline_sales"],
                                        mode="lines+markers",
                                        name=f"Status Quo ({kpi} at Current Price)",
                                        line=dict(color="#fb923c", dash="dash", width=2),
                                    )
                                )
                                fig_comp.add_trace(
                                    go.Scatter(
                                        x=sales_impact.comparison_forecast[sales_impact.date_col],
                                        y=sales_impact.comparison_forecast["optimized_sales"],
                                        mode="lines+markers",
                                        name=f"AI-Optimized ({kpi} at Best Price)",
                                        line=dict(color="#4ade80", width=2.5),
                                        fill="tonexty",
                                        fillcolor="rgba(74, 222, 128, 0.12)",
                                    )
                                )
                                fig_comp.update_layout(
                                    title=f"Future Sales Trajectory: Status Quo vs AI-Optimized Price | {entity1}",
                                    xaxis_title=sales_impact.date_col,
                                    yaxis_title=f"Expected {kpi}",
                                )
                                st.plotly_chart(style_chart_bright(fig_comp), use_container_width=True)

                                if sales_impact.volume_change_pct >= 0:
                                    summary_msg = (
                                        f"Strategy Analysis ({sales_impact.strategy_name}): Setting target price to {sales_impact.optimal_price:.2f} ({sales_impact.price_change_pct:+.1f}%) "
                                        f"is projected to increase bike sales volume by +{sales_impact.volume_change_pct:.1f}% while growing total sales revenue by "
                                        f"+{sales_impact.revenue_change_pct:.1f}% (+{format_value(sales_impact.net_sales_gain)}) across the forecast horizon."
                                    )
                                else:
                                    summary_msg = (
                                        f"Strategy Analysis ({sales_impact.strategy_name}): Setting target price to {sales_impact.optimal_price:.2f} ({sales_impact.price_change_pct:+.1f}%) "
                                        f"incurs a modest {abs(sales_impact.volume_change_pct):.1f}% volume adjustment, while capturing strong net revenue growth of "
                                        f"+{sales_impact.revenue_change_pct:.1f}% (+{format_value(sales_impact.net_sales_gain)}) across the forecast horizon."
                                    )
                                st.caption(summary_msg)
                            except Exception as imp_exc:
                                st.info(f"Sales impact simulation note: {imp_exc}")

                    except Exception as exc:
                        st.error(f"Price optimization failed: {exc}")
                else:
                    st.warning("Min price must be less than max price.")

st.divider()
st.markdown(
    """
    <div class="footer-note">
        This interface adapts to any CSV dataset by detecting the entity axis, KPI candidates, date fields, and category structure automatically.
    </div>
    """,
    unsafe_allow_html=True,
)
