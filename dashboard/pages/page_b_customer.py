"""
dashboard/pages/page_b_customer.py — Customer 360 Intelligence
"""
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.kpis import customer_360


def render(data: dict):
    st.markdown("""
    <div class="main-header">
        <h1>👤 Customer Intelligence — 360° View</h1>
        <p>Customer profile, observed engagement signals, product ownership, and review actions</p>
    </div>
    """, unsafe_allow_html=True)

    feat = data.get("features")
    if feat is None:
        st.error("Feature data not loaded.")
        return

    # ── Customer Selector ─────────────────────────────────────────────
    col_search, col_segment = st.columns([2, 1])
    with col_search:
        customer_ids = sorted(feat["customer_id"].unique().tolist())
        selected_id = st.selectbox(
            "Search Customer ID",
            options=customer_ids,
            index=0,
            help="Select or type a customer ID",
        )
    with col_segment:
        seg = data.get("segments")
        if seg is not None:
            seg_row = seg[seg["customer_id"] == selected_id]
            if not seg_row.empty:
                st.metric("Segment", seg_row.iloc[0]["segment_label"])

    if selected_id is None:
        return

    profile = customer_360(selected_id, data)

    # ── Top-line customer KPIs ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    feat_row = profile.get("features", {})
    risk_info = profile.get("risk_indicator", {})
    nba_info = profile.get("nba", {})

    with c1:
        score = risk_info.get("risk_indicator_score", "N/A")
        band = risk_info.get("risk_indicator_band")
        color = "#ef5350" if band == "upper" else (
            "#ff9800" if band == "middle" else "#66bb6a"
        )
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color}">{score if isinstance(score, str) else f"{score:.3f}"}</div>
            <div class="kpi-label">Relative Review Indicator</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:#4fc3f7">{feat_row.get('frequency_total', 'N/A')}</div>
            <div class="kpi-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        mon = feat_row.get("monetary_total", 0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:#81c784">{mon/1e6:.1f}M</div>
            <div class="kpi-label">Transaction Amount (million source units)</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        sat = feat_row.get("satisfaction_score", "N/A")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:#ce93d8">{sat}/10</div>
            <div class="kpi-label">Satisfaction</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        eng = feat_row.get("engagement_score", 0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:#ffb74d">{eng:.2f}</div>
            <div class="kpi-label">Engagement Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Profile & RFM ────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
        cust_profile = profile.get("profile", {})
        profile_fields = {
            "Gender": cust_profile.get("gender", "N/A"),
            "Age Group": feat_row.get("age_group", "N/A"),
            "Income Bracket": cust_profile.get("income_bracket", "N/A"),
            "Occupation": cust_profile.get("occupation", "N/A"),
            "Education": cust_profile.get("education_level", "N/A"),
            "Marital Status": cust_profile.get("marital_status", "N/A"),
            "Household Size": cust_profile.get("household_size", "N/A"),
            "Tenure (months)": feat_row.get("customer_tenure_months", "N/A"),
            "Acquisition": cust_profile.get("acquisition_channel", "N/A"),
        }
        for label, value in profile_fields.items():
            val_str = f"{value:.1f}" if isinstance(value, float) else str(value)
            st.markdown(f"**{label}:** {val_str}")

    with right:
        st.markdown('<div class="section-title">RFM Analysis</div>', unsafe_allow_html=True)
        rfm_data = {
            "Recency (days)": feat_row.get("recency_days", 0),
            "Frequency (total)": feat_row.get("frequency_total", 0),
            "Average transaction amount (source units)": feat_row.get("monetary_avg", 0),
        }
        fig = go.Figure(go.Scatterpolar(
            r=[rfm_data["Recency (days)"], rfm_data["Frequency (total)"],
               rfm_data["Average transaction amount (source units)"] / 1e6],
            theta=["Recency (days)", "Frequency (transactions)", "Average amount (million source units)"],
            fill="toself",
            line=dict(color="#4fc3f7"),
            fillcolor="rgba(79,195,247,0.2)",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#e0e0e0"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#e0e0e0"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            height=300,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Product Propensity ────────────────────────────────────────────
    st.markdown('<div class="section-title">Product Ownership</div>', unsafe_allow_html=True)
    ownership_info = profile.get("product_ownership", {})
    if ownership_info:
        products = ["credit_card", "personal_loan", "investment_account", "insurance_product"]
        prop_cols = st.columns(len(products))
        for i, prod in enumerate(products):
            owned = ownership_info.get(f"{prod}_owned", None)
            with prop_cols[i]:
                if owned is not None:
                    status = "Owned" if owned else "Not currently owned"
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-value" style="color:#4fc3f7">{status}</div>
                        <div class="kpi-label">{prod.replace('_', ' ').title()}</div>
                        <div style="font-size:0.75rem;color:#aaa;margin-top:4px">Current dataset fact only</div>
                    </div>
                    """, unsafe_allow_html=True)
    st.caption(
        "No future product-uptake labels, eligibility data, customer needs, or consent fields "
        "are available; no product propensity or offer recommendation is made."
    )

    # ── NBA Recommendation ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Next-Best-Action Recommendation</div>', unsafe_allow_html=True)
    if nba_info:
        action = nba_info.get("action_description", "No recommendation")
        reason = nba_info.get("reason", "")
        evidence = nba_info.get("evidence", "[]")
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except Exception:
                evidence = [evidence]

        st.markdown(f"""
        <div class="action-card">
            <div class="action-title">🔮 {action}</div>
            <div class="action-reason">{reason}</div>
            <div style="font-size:0.75rem;color:#888;margin-top:0.5rem">
                Evidence: {', '.join(evidence) if isinstance(evidence, list) else evidence}
            </div>
            <div style="font-size:0.7rem;color:#ff9800;margin-top:0.3rem">
                ⚠️ Requires human review before any customer action
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Transaction Summary ───────────────────────────────────────────
    tx_summary = profile.get("transaction_summary", {})
    if tx_summary:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Transaction Breakdown</div>', unsafe_allow_html=True)
        tx_df = pd.DataFrame(tx_summary).T
        tx_df.columns = ["Count", "Total Volume", "Avg Amount"]
        tx_df = tx_df.round(0).astype(int)
        st.dataframe(tx_df, use_container_width=True)

    # ── Recent Transactions ───────────────────────────────────────────
    recent = profile.get("recent_transactions", [])
    if recent:
        st.markdown('<div class="section-title">Recent Transactions</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
