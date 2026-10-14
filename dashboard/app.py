"""
dashboard/app.py
─────────────────────────────────────────────────────────────
AI-Powered Banking Customer Intelligence Platform
Streamlit Dashboard — Main Entry Point

Pages:
  A: Executive Overview
  B: Customer Intelligence (360 Profile)
  C: Engagement Review
  D: Product & Segment Analytics
  E: Next-Best-Action Management

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html {
    font-size: 18px;
}

body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Gradient header */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 1.05rem; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(145deg, #1e2a3a, #162032);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card .kpi-value { font-size: 2.1rem; font-weight: 700; color: #4fc3f7; }
.kpi-card .kpi-label { font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 1px; }
.kpi-card .kpi-delta { font-size: 0.95rem; margin-top: 0.2rem; }

/* Risk badges */
.badge-high   { background: #ef5350; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }
.badge-medium { background: #ff9800; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }
.badge-low    { background: #66bb6a; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }

/* Section titles */
.section-title {
    font-size: 1.3rem; font-weight: 600;
    color: #e0e0e0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #4fc3f7;
    margin-bottom: 1rem;
}

/* Disclaimer box */
.disclaimer {
    background: rgba(255,152,0,0.1);
    border-left: 4px solid #ff9800;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 1rem;
    color: #ffcc80;
    margin: 1rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size: 1.05rem;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    font-size: 0.9rem;
}

/* Action card */
.action-card {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    border: 1px solid rgba(79,195,247,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.action-card .action-title { font-size: 1.1rem; font-weight: 600; color: #4fc3f7; }
.action-card .action-reason { font-size: 1rem; color: rgba(255,255,255,0.8); margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data():
    from src.analytics.kpis import load_all_data
    return load_all_data()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <span style="font-size:2.5rem;">🏦</span>
        <h3 style="color:#4fc3f7;margin:0.5rem 0;">Banking Intel</h3>
        <p style="color:rgba(255,255,255,0.5);font-size:0.8rem;">Customer Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate to",
        options=[
            "Executive Overview",
            "Customer Intelligence",
            "Engagement Review",
            "Products & Segments",
            "Next-Best-Action",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Data: COFINFAD 2023 | Segmentation: K-Means | Decision rules: rule_v2")
    st.caption("⚠️ No validated churn or product-uptake model is available")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
data = load_data()

# ─────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────
if page == "Executive Overview":
    from dashboard.pages.page_a_overview import render
    render(data)

elif page == "Customer Intelligence":
    from dashboard.pages.page_b_customer import render
    render(data)

elif page == "Engagement Review":
    from dashboard.pages.page_c_churn import render
    render(data)

elif page == "Products & Segments":
    from dashboard.pages.page_d_products import render
    render(data)

elif page == "Next-Best-Action":
    from dashboard.pages.page_e_nba import render
    render(data)
