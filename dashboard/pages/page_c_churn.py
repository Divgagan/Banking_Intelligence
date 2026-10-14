"""
dashboard/pages/page_c_churn.py — Churn Analytics
"""
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.kpis import engagement_indicator_analytics


def render(data: dict):
    st.markdown("""
    <div class="main-header">
        <h1>⚠️ Customer Engagement Review</h1>
        <p>Transparent review indicators and the evidence limits of the available data</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ No churn model:</strong> The dataset has no verified churn outcome or prediction window.
        The precomputed source churn score is undocumented and is not used as a label. The indicator
        shown here ranks observed engagement signals; it is not a churn probability.
    </div>
    """, unsafe_allow_html=True)

    # ── Model Performance ─────────────────────────────────────────────
    report_path = Path("reports/churn_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        st.caption(f"Score status: {report.get('status', 'unknown')}")
        st.info(
            "Model metrics and SHAP values are intentionally not shown: no independently "
            "defined churn outcome is available for valid supervised evaluation."
        )

    # ── Risk Distribution ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    analytics = engagement_indicator_analytics(data)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Relative Review Bands</div>', unsafe_allow_html=True)
        tiers = analytics.get("risk_tiers", {})
        if tiers:
            fig = go.Figure(go.Pie(
                labels=["Upper", "Middle", "Lower"],
                values=[tiers.get("upper", 0), tiers.get("middle", 0), tiers.get("lower", 0)],
                marker=dict(colors=["#ef5350", "#ff9800", "#66bb6a"]),
                hole=0.45,
                textinfo="label+percent",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                height=350,
                margin=dict(t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Average Review Indicator by Segment</div>', unsafe_allow_html=True)
        churn_seg = analytics.get("churn_by_segment", {})
        if churn_seg:
            seg_df = pd.DataFrame(churn_seg).T
            seg_df = seg_df.reset_index().rename(columns={"index": "Segment", "mean": "Avg Indicator", "count": "Customers"})
            fig = px.bar(
                seg_df, x="Segment", y="Avg Indicator",
                color="Avg Indicator",
                color_continuous_scale=["#66bb6a", "#ff9800", "#ef5350"],
                text=seg_df["Avg Indicator"].apply(lambda x: f"{x:.3f}"),
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                height=350,
                margin=dict(t=20, b=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── High-risk customer table ──────────────────────────────────────
    st.markdown('<div class="section-title">Customers with Highest Relative Review Indicators</div>', unsafe_allow_html=True)
    churn_df = data.get("risk_indicator")
    feat_df  = data.get("features")
    if churn_df is not None and feat_df is not None:
        merged = churn_df.merge(
            feat_df[["customer_id", "frequency_total", "monetary_total",
                      "satisfaction_score", "engagement_score"]],
            on="customer_id", how="left"
        )
        top20 = merged.nlargest(20, "risk_indicator_score")[[
            "customer_id", "risk_indicator_score", "frequency_total",
            "monetary_total", "satisfaction_score", "engagement_score"
        ]].rename(columns={
            "monetary_total": "amount_millions_source_units",
        })
        top20["amount_millions_source_units"] = top20[
            "amount_millions_source_units"
        ].apply(lambda x: f"{x/1e6:.1f}M")
        top20["risk_indicator_score"] = top20["risk_indicator_score"].apply(lambda x: f"{x:.3f}")
        st.dataframe(top20, use_container_width=True, hide_index=True)

    # ── Limitations ───────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        <strong>Indicator Limitations:</strong>
        <ul style="margin:0.3rem 0;padding-left:1.2rem;">
            <li>Values are relative ranks of observed activity/satisfaction signals, not probabilities</li>
            <li>Dataset spans about one year and has no verified future churn outcome</li>
            <li>No calibration, model-performance, or causal retention claim is supported</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
