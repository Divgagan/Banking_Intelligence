"""
dashboard/pages/page_a_overview.py — Executive Overview
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.kpis import executive_kpis


def _kpi_card(label: str, value: str, delta: str = "", color: str = "#4fc3f7"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
        {f'<div class="kpi-delta" style="color:#66bb6a">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def render(data: dict):
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Executive Overview</h1>
        <p>AI-Powered Banking Customer Intelligence Platform — dataset snapshot overview</p>
    </div>
    """, unsafe_allow_html=True)

    kpis = executive_kpis(data)

    # ── Row 1: Top-line KPIs ─────────────────────────────────────────
    cols = st.columns(5)
    with cols[0]:
        _kpi_card("Total Customers", f"{kpis.get('total_customers', 0):,}", color="#4fc3f7")
    with cols[1]:
        _kpi_card("Total Transactions", f"{kpis.get('total_transactions', 0):,}", color="#81c784")
    with cols[2]:
        amount = kpis.get('total_transaction_amount', 0)
        _kpi_card("Transaction Amount (source units)", f"{amount / 1e9:.1f}B", color="#ffb74d")
    with cols[3]:
        _kpi_card("Avg Satisfaction", f"{kpis.get('avg_satisfaction', 0):.1f}/10", color="#ce93d8")
    with cols[4]:
        _kpi_card("Upper Review Band", f"{kpis.get('upper_indicator_band', 0):,}", color="#ef5350")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Segment & Product mix ──────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Customer Segments</div>', unsafe_allow_html=True)
        seg_dist = kpis.get("segment_dist", {})
        if seg_dist:
            fig = px.pie(
                names=list(seg_dist.keys()),
                values=list(seg_dist.values()),
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Tealgrn,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                showlegend=True,
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Product Ownership (%)</div>', unsafe_allow_html=True)
        products = ["savings_account", "credit_card", "personal_loan",
                     "investment_account", "insurance_product"]
        product_pcts = {p.replace("_", " ").title(): kpis.get(f"{p}_pct", 0) for p in products}
        fig = go.Figure(go.Bar(
            x=list(product_pcts.values()),
            y=list(product_pcts.keys()),
            orientation="h",
            marker=dict(
                color=list(product_pcts.values()),
                colorscale="Viridis",
                line=dict(width=0),
            ),
            text=[f"{v}%" for v in product_pcts.values()],
            textposition="auto",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            xaxis_title="Ownership %",
            height=350,
            margin=dict(t=20, b=40, l=20, r=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Churn Risk & Demographics ──────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Exploratory Engagement Review Indicator</div>', unsafe_allow_html=True)
        churn_data = data.get("risk_indicator")
        if churn_data is not None and "risk_indicator_score" in churn_data:
            fig = px.histogram(
                churn_data, x="risk_indicator_score",
                nbins=40,
                color_discrete_sequence=["#ef5350"],
                labels={"risk_indicator_score": "Relative Review Indicator (not a probability)"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                height=320,
                margin=dict(t=20, b=40, l=20, r=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis_title="Customer Count",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Income Distribution</div>', unsafe_allow_html=True)
        income_dist = kpis.get("income_dist", {})
        if income_dist:
            fig = px.bar(
                x=list(income_dist.keys()),
                y=list(income_dist.values()),
                color=list(income_dist.keys()),
                color_discrete_sequence=px.colors.sequential.Plasma,
                labels={"x": "Income Bracket", "y": "Customers"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                height=320,
                showlegend=False,
                margin=dict(t=20, b=40, l=20, r=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Transaction trends ─────────────────────────────────────
    st.markdown('<div class="section-title">Monthly Transaction Amount (currency unspecified)</div>', unsafe_allow_html=True)
    tx_month = kpis.get("tx_by_month", {})
    if tx_month:
        months = sorted(tx_month.keys())
        amounts = [tx_month[m]["amount"] for m in months]
        counts  = [tx_month[m]["tx_count"] for m in months]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=months, y=amounts, name="Transaction amount (source units)",
            marker_color="#4fc3f7", opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=months, y=counts, name="Tx Count",
            yaxis="y2", line=dict(color="#ffb74d", width=3),
            mode="lines+markers",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            height=350,
            margin=dict(t=20, b=40, l=60, r=60),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Amount (source units)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis2=dict(title="Transaction Count", overlaying="y", side="right",
                        gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Disclaimer ────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Data Limitations:</strong> No verified churn outcomes or future product-uptake labels
        are available. The relative engagement indicator is not a probability. Product ownership is
        reported as a historical fact, not a recommendation. All suggested actions require human review.
    </div>
    """, unsafe_allow_html=True)
