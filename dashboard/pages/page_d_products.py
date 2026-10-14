"""
dashboard/pages/page_d_products.py — Sales & Product Analytics
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.kpis import segment_summary


def render(data: dict):
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Products & Segments</h1>
        <p>Segment profiles and observed product ownership; no uptake prediction is available</p>
    </div>
    """, unsafe_allow_html=True)

    feat = data.get("features")
    seg  = data.get("segments")

    # ── Segment Profiling ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Segment Profiling</div>', unsafe_allow_html=True)

    seg_summ = segment_summary(data)
    if not seg_summ.empty:
        # Format for display
        display_df = seg_summ.copy()
        rename_map = {
            "customer_count": "Customers",
            "recency_days": "Avg Recency",
            "frequency_total": "Avg Frequency",
            "monetary_avg": "Avg Tx Amount",
            "monetary_total": "Avg Total Volume",
            "engagement_score": "Avg Engagement",
            "satisfaction_score": "Avg Satisfaction",
            "nps_score": "Avg NPS",
            "active_products": "Avg Products",
        }
        if "risk_indicator_score" in display_df.columns:
            rename_map["risk_indicator_score"] = "Avg Review Indicator"

        display_df = display_df.rename(columns=rename_map)
        st.dataframe(display_df, use_container_width=True)

        # ── Radar chart comparing segments ────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Segment Comparison (Radar)</div>', unsafe_allow_html=True)

        radar_cols = ["frequency_total", "engagement_score", "satisfaction_score",
                      "active_products", "nps_score"]
        available_radar = [c for c in radar_cols if c in seg_summ.columns]

        if available_radar:
            fig = go.Figure()
            colors = ["#4fc3f7", "#ef5350", "#81c784", "#ffb74d", "#ce93d8"]
            for i, (seg_name, row) in enumerate(seg_summ.iterrows()):
                vals = [row[c] for c in available_radar]
                # Normalize to 0-1 range for comparable radar
                max_vals = seg_summ[available_radar].max()
                norm_vals = [(v / mx if mx > 0 else 0) for v, mx in zip(vals, max_vals)]
                norm_vals.append(norm_vals[0])  # close the polygon

                fig.add_trace(go.Scatterpolar(
                    r=norm_vals,
                    theta=[c.replace("_", " ").title() for c in available_radar] + [available_radar[0].replace("_", " ").title()],
                    fill="toself",
                    name=str(seg_name),
                    line=dict(color=colors[i % len(colors)]),
                    fillcolor=f"rgba({','.join(str(int(c, 16)) for c in [colors[i % len(colors)][1:3], colors[i % len(colors)][3:5], colors[i % len(colors)][5:7]])},0.1)",
                ))

            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1.1], gridcolor="rgba(255,255,255,0.1)", color="#aaa"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#e0e0e0"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                height=450,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Product Ownership by Segment ──────────────────────────────────
    st.markdown('<div class="section-title">Product Ownership by Segment</div>', unsafe_allow_html=True)

    if feat is not None and seg is not None:
        merged = feat.merge(seg[["customer_id", "segment_label"]], on="customer_id", how="left")
        products = ["savings_account", "credit_card", "personal_loan",
                     "investment_account", "insurance_product"]
        available_prods = [p for p in products if p in merged.columns]

        if available_prods:
            ownership_by_seg = merged.groupby("segment_label")[available_prods].mean() * 100
            ownership_by_seg = ownership_by_seg.round(1)

            fig = px.imshow(
                ownership_by_seg.values,
                x=[p.replace("_", " ").title() for p in available_prods],
                y=ownership_by_seg.index.tolist(),
                color_continuous_scale="Viridis",
                labels=dict(color="Ownership %"),
                text_auto=".1f",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
