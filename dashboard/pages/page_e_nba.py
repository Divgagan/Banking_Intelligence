"""
dashboard/pages/page_e_nba.py — Next-Best-Action Management
"""
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.kpis import nba_summary


def render(data: dict):
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Next-Best-Action Command Center</h1>
        <p>Rule-based recommendations with full transparency, evidence, and human-in-the-loop review</p>
    </div>
    """, unsafe_allow_html=True)

    nba = data.get("nba")
    if nba is None:
        st.error("NBA recommendations not loaded. Run: python -m src.recommendations.nba_engine")
        return

    summary = nba_summary(data)

    # ── Action Distribution KPIs ──────────────────────────────────────
    action_dist = summary.get("action_distribution", {})
    if action_dist:
        # Color map for actions
        action_colors = {
            "ENGAGEMENT_REVIEW": "#ef5350",
            "REENGAGEMENT": "#ffb74d",
            "SERVICE_REVIEW": "#ce93d8",
            "MONITOR": "#9e9e9e",
        }

        total = sum(action_dist.values())
        cols = st.columns(min(len(action_dist), 4))
        sorted_actions = sorted(action_dist.items(), key=lambda x: x[1], reverse=True)

        for i, (action, count) in enumerate(sorted_actions[:4]):
            with cols[i]:
                color = action_colors.get(action, "#9e9e9e")
                pct = count / total * 100
                st.markdown(f"""
                <div class="kpi-card" style="border-left: 3px solid {color}">
                    <div class="kpi-value" style="color:{color}">{count:,}</div>
                    <div class="kpi-label">{action.replace('_', ' ')}</div>
                    <div style="font-size:0.8rem;color:#aaa">{pct:.1f}% of all</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action Distribution Chart ─────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Action Type Distribution</div>', unsafe_allow_html=True)
        if action_dist:
            fig = go.Figure(go.Pie(
                labels=[a.replace("_", " ").title() for a in action_dist.keys()],
                values=list(action_dist.values()),
                hole=0.45,
                marker=dict(colors=[action_colors.get(a, "#9e9e9e") for a in action_dist.keys()]),
                textinfo="label+percent",
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                height=400,
                margin=dict(t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Actions by Segment</div>', unsafe_allow_html=True)
        seg_dist = summary.get("segment_distribution", [])
        if seg_dist:
            seg_df = pd.DataFrame(seg_dist)
            if not seg_df.empty:
                fig = px.bar(
                    seg_df, x="segment_label", y="count", color="action_code",
                    color_discrete_map=action_colors,
                    barmode="stack",
                    labels={"count": "Customers", "segment_label": "Segment", "action_code": "Action"},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0e0",
                    height=400,
                    margin=dict(t=20, b=20),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    legend=dict(font_size=9),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Filterable Action Queue ───────────────────────────────────────
    st.markdown('<div class="section-title">📋 Action Queue (Filterable)</div>', unsafe_allow_html=True)

    filter_cols = st.columns(3)
    with filter_cols[0]:
        action_filter = st.multiselect(
            "Filter by Action",
            options=sorted(nba["action_code"].unique()),
            default=["ENGAGEMENT_REVIEW"] if "ENGAGEMENT_REVIEW" in nba["action_code"].values else [],
        )
    with filter_cols[1]:
        seg_filter = st.multiselect(
            "Filter by Segment",
            options=sorted(nba["segment_label"].unique()) if "segment_label" in nba.columns else [],
        )
    with filter_cols[2]:
        priority_range = st.slider("Min Priority Score", 0, 100, 60)

    # Apply filters
    filtered = nba.copy()
    if action_filter:
        filtered = filtered[filtered["action_code"].isin(action_filter)]
    if seg_filter:
        filtered = filtered[filtered["segment_label"].isin(seg_filter)]
    filtered = filtered[filtered["priority_score"] >= priority_range]

    # Sort by priority
    filtered = filtered.sort_values("priority_score", ascending=False)

    st.markdown(f"**Showing {len(filtered):,} of {len(nba):,} customers**")

    # Display table
    display_cols = ["customer_id", "segment_label", "risk_indicator_score", "action_code",
                    "action_description", "reason", "priority_score"]
    available_display = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[available_display].head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "customer_id": "Customer ID",
            "segment_label": "Segment",
            "risk_indicator_score": st.column_config.ProgressColumn(
                "Relative Review Indicator", format="%.2f", min_value=0, max_value=1
            ),
            "action_code": "Action Code",
            "action_description": "Recommendation",
            "reason": "Reason",
            "priority_score": st.column_config.NumberColumn("Priority", format="%d"),
        },
    )

    # ── Detail view for selected customer ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Action Detail</div>', unsafe_allow_html=True)

    if not filtered.empty:
        selected = st.selectbox(
            "Select customer for detail",
            options=filtered["customer_id"].head(20).tolist(),
        )
        if selected:
            row = nba[nba["customer_id"] == selected].iloc[0]
            evidence = row.get("evidence", "[]")
            if isinstance(evidence, str):
                try:
                    evidence = json.loads(evidence)
                except json.JSONDecodeError:
                    evidence = [evidence]

            all_actions = row.get("all_actions_json", "[]")
            if isinstance(all_actions, str):
                try:
                    all_actions = json.loads(all_actions)
                except json.JSONDecodeError:
                    all_actions = []

            st.markdown(f"""
            <div class="action-card">
                <div class="action-title">🔮 {row['action_description']}</div>
                <div class="action-reason">{row['reason']}</div>
                <div style="margin-top:0.5rem;font-size:0.82rem;">
                    <strong style="color:#4fc3f7">Evidence:</strong>
                    <ul style="margin:0.2rem 0;padding-left:1.2rem;color:#aaa;">
                        {''.join(f'<li>{e}</li>' for e in evidence)}
                    </ul>
                </div>
                <div style="font-size:0.7rem;color:#ff9800;margin-top:0.5rem;">
                    ⚠️ This is an informational rule-based suggestion — requires human review before action.
                    Priority score: {row['priority_score']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if all_actions and len(all_actions) > 1:
                st.markdown("**Alternative actions:**")
                for alt in all_actions[1:]:
                    st.markdown(f"- **{alt.get('action_code', '')}**: {alt.get('reason', '')}")

    # ── Engine metadata ───────────────────────────────────────────────
    with st.expander("🛠️ Engine Configuration & Rules"):
        st.markdown("""
        | Rule | Threshold | Priority |
        |------|-----------|----------|
        | Engagement review | Relative indicator ≥ 0.667 | 80 |
        | Recent activity review | No transaction for ≥ 60 days | 70 |
        | Service review | Satisfaction ≤ 3 or complaint signal | 60 |
        | Monitor | No available signal exceeds threshold | 0 |
        """)

        st.markdown("""
        <div class="disclaimer">
            <strong>Limitations:</strong>
            <ul>
                <li>Rules are deterministic — no ML-based action ranking</li>
                <li>Engagement indicator is not a churn probability</li>
                <li>No product offers are generated without uptake labels and eligibility evidence</li>
                <li>No A/B testing or causal uplift modelling</li>
                <li>All recommendations are informational and require human review</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
