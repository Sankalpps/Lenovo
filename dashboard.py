"""
NVMe Drive Health Intelligence Dashboard
==========================================
Premium dashboard — unique flowing layout, glassmorphism,
animated elements, Lenovo branding.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_generator import load_dataset, FAILURE_MODE_MAP
from data_processing import prepare_features, scale_features
from ml_model import (
    split_data, train_random_forest, evaluate_model,
    get_feature_importance, predict_failure_probability
)
from intelligence_engine import analyze_all_drives

# ━━━━━━━━━━━━━━━ PAGE CONFIG ━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="NVMe Health Intelligence | Lenovo",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━ FULL CSS — UNIQUE THEME ━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #06090f;
    --bg-secondary: #0c1018;
    --bg-card: rgba(14,19,29,0.85);
    --border: rgba(255,255,255,0.04);
    --border-hover: rgba(226,35,26,0.25);
    --accent: #E2231A;
    --accent-glow: rgba(226,35,26,0.12);
    --text-primary: #eaeef3;
    --text-secondary: #7a8599;
    --text-muted: #4a5568;
    --green: #34d399;
    --yellow: #fbbf24;
    --red: #f87171;
    --blue: #60a5fa;
    --purple: #a78bfa;
}

*, *::before, *::after { font-family: 'Inter', -apple-system, sans-serif !important; }

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(226,35,26,0.06), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(96,165,250,0.03), transparent);
}

/* Hide all Streamlit chrome */
#MainMenu, header, footer, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── HERO BANNER ── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #0c0f16 0%, #131824 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 32px;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(circle at 15% 50%, rgba(226,35,26,0.12) 0%, transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(96,165,250,0.06) 0%, transparent 40%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    top: -1px; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent 5%, var(--accent) 30%, var(--accent) 70%, transparent 95%);
    opacity: 0.7;
}
.hero-flex {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 1;
}
.hero-left h1 {
    font-size: 28px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.8px;
    margin: 0 0 6px;
    line-height: 1.2;
}
.hero-left h1 span { color: var(--accent); }
.hero-left p {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
}
.hero-right {
    display: flex;
    gap: 12px;
    align-items: center;
}
.hero-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    padding: 8px 16px;
    border-radius: 10px;
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 500;
    backdrop-filter: blur(8px);
}
.hero-chip .val {
    color: var(--text-primary);
    font-weight: 700;
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── METRIC RING ── */
.ring-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 36px;
}
.ring-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.ring-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 0 30px var(--accent-glow);
}
.ring-num {
    font-size: 34px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -1.5px;
    line-height: 1;
    margin: 0 0 8px;
}
.ring-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.8px;
}
.ring-sub {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 6px;
    font-weight: 400;
}
.c-red    { color: var(--accent); }
.c-green  { color: var(--green); }
.c-yellow { color: var(--yellow); }
.c-blue   { color: var(--blue); }
.c-purple { color: var(--purple); }

/* ── SECTION HEADER ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 40px 0 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}
.sec-head h2 {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.3px;
}
.sec-head .tag {
    background: rgba(226,35,26,0.1);
    color: var(--accent);
    font-size: 10px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── PANEL (glassmorphism card) ── */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease;
}
.panel:hover {
    border-color: rgba(255,255,255,0.06);
}
.panel-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 18px;
}

/* ── STAT BAR ── */
.stat-bar {
    display: flex;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.02);
}
.stat-bar:last-child { border-bottom: none; }
.stat-bar .name {
    flex: 1;
    font-size: 13px;
    color: var(--text-secondary);
    font-weight: 500;
}
.stat-bar .bar-track {
    flex: 2;
    height: 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 3px;
    margin: 0 16px;
    overflow: hidden;
}
.stat-bar .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.stat-bar .val {
    font-size: 13px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary);
    min-width: 52px;
    text-align: right;
}

/* ── DRIVE TABLE ── */
.stDataFrame { border-radius: 14px; overflow: hidden; }

/* ── TABS (minimal) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 12px 20px;
    font-weight: 600;
    font-size: 13px;
    color: var(--text-muted);
    border-bottom: 2px solid transparent;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ── FOOTER ── */
.foot {
    text-align: center;
    padding: 32px 0 8px;
    margin-top: 48px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 11px;
    letter-spacing: 0.5px;
}
.foot b { color: var(--accent); font-weight: 700; }

/* ── Streamlit element overrides ── */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
}
div[data-testid="stMetric"] label { color: var(--text-muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━ DATA ━━━━━━━━━━━━━━━
@st.cache_data(show_spinner=False)
def load_data():
    df = load_dataset("data/nvme_dataset.csv")
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    model = train_random_forest(X_train_s, y_train)
    metrics, y_pred, y_proba = evaluate_model(model, X_test_s, y_test, "Random Forest")
    importance_df = get_feature_importance(model, feature_names)
    X_all_s = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    failure_probs = predict_failure_probability(model, X_all_s)
    intel_results = analyze_all_drives(df, failure_probs)
    df["health_score"] = intel_results["health_score"].values
    df["status"] = intel_results["status"].values
    df["failure_probability"] = intel_results["failure_probability"].values
    df["reasons"] = intel_results["reasons"].values
    df["recommendations"] = intel_results["recommendations"].values
    return df, metrics, importance_df, intel_results

with st.spinner(""):
    df, metrics, importance_df, intel_results = load_data()


# ━━━━━━━━━━━━━━━ PLOTLY THEME ━━━━━━━━━━━━━━━
PLT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#7a8599", size=11),
    margin=dict(l=0, r=0, t=32, b=0),
    hoverlabel=dict(bgcolor="#131824", font_size=12, font_family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.02)", zerolinecolor="rgba(255,255,255,0.02)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.02)", zerolinecolor="rgba(255,255,255,0.02)"),
)

STATUS_C = {"Healthy": "#34d399", "Warning": "#fbbf24", "Critical": "#f87171"}
MODE_C   = {"None": "#60a5fa", "Wear-Out": "#f87171", "Thermal": "#fb923c",
            "Power": "#a78bfa", "Firmware": "#38bdf8", "Early-Life": "#fbbf24"}


# ━━━━━━━━━━━━━━━ SIDEBAR ━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("<div style='padding:12px 0; text-align:center;'>"
                "<span style='font-size:22px; font-weight:800; color:#E2231A;'>Lenovo</span>"
                "<br/><span style='font-size:10px; color:#4a5568; letter-spacing:2px; text-transform:uppercase;'>NVME INTELLIGENCE</span>"
                "</div>", unsafe_allow_html=True)
    st.markdown("---")
    vendors  = st.multiselect("Vendor",  sorted(df["Vendor"].unique()),  default=sorted(df["Vendor"].unique()))
    models   = st.multiselect("Model",   sorted(df["Model"].unique()),   default=sorted(df["Model"].unique()))
    modes    = st.multiselect("Failure Mode", sorted(df["Failure_Mode_Label"].unique()), default=sorted(df["Failure_Mode_Label"].unique()))
    statuses = st.multiselect("Status",  ["Healthy","Warning","Critical"], default=["Healthy","Warning","Critical"])
    score_range = st.slider("Health Score Range", 0, 100, (0, 100))

mask = (
    df["Vendor"].isin(vendors) & df["Model"].isin(models) &
    df["Failure_Mode_Label"].isin(modes) & df["status"].isin(statuses) &
    df["health_score"].between(score_range[0], score_range[1])
)
fdf = df[mask].copy()

total    = len(fdf)
failed   = int(fdf["Failure_Flag"].sum())
fr       = (failed / total * 100) if total else 0
avg_h    = fdf["health_score"].mean() if total else 0
crit_n   = int((fdf["status"] == "Critical").sum())
warn_n   = int((fdf["status"] == "Warning").sum())
healthy_n = int((fdf["status"] == "Healthy").sum())


# ━━━━━━━━━━━━━━━ HERO ━━━━━━━━━━━━━━━
st.markdown(f"""
<div class="hero">
  <div class="hero-flex">
    <div class="hero-left">
      <h1>NVMe Drive <span>Health Intelligence</span></h1>
      <p>Predictive failure analysis & smart maintenance — {total:,} drives under monitoring</p>
    </div>
    <div class="hero-right">
      <div class="hero-chip">Fleet <div class="val">{total:,}</div></div>
      <div class="hero-chip">Failed <div class="val c-red">{failed}</div></div>
      <div class="hero-chip">Accuracy <div class="val c-green">{metrics['accuracy']:.0f}%</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━ KPI RINGS ━━━━━━━━━━━━━━━
st.markdown(f"""
<div class="ring-grid">
  <div class="ring-card">
    <div class="ring-num c-red">{total:,}</div>
    <div class="ring-label">Total Drives</div>
    <div class="ring-sub">{len(df['Vendor'].unique())} vendors &middot; {len(df['Model'].unique())} models</div>
  </div>
  <div class="ring-card">
    <div class="ring-num c-yellow">{fr:.1f}%</div>
    <div class="ring-label">Failure Rate</div>
    <div class="ring-sub">{failed} of {total} drives failed</div>
  </div>
  <div class="ring-card">
    <div class="ring-num c-green">{avg_h:.1f}</div>
    <div class="ring-label">Health Score</div>
    <div class="ring-sub">fleet average (0-100)</div>
  </div>
  <div class="ring-card">
    <div class="ring-num c-blue">{metrics['accuracy']:.1f}%</div>
    <div class="ring-label">ML Accuracy</div>
    <div class="ring-sub">F1: {metrics['f1_score']:.1f}% &middot; RF 200 trees</div>
  </div>
  <div class="ring-card">
    <div class="ring-num c-purple">{crit_n + warn_n}</div>
    <div class="ring-label">Active Alerts</div>
    <div class="ring-sub">{crit_n} critical &middot; {warn_n} warning</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━ STATUS SNAPSHOT ━━━━━━━━━━━━━━━
st.markdown('<div class="sec-head"><h2>Fleet Health Snapshot</h2><span class="tag">Live</span></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 3, 3])

with col1:
    # Donut
    fig_d = go.Figure(go.Pie(
        labels=["Healthy","Warning","Critical"], values=[healthy_n, warn_n, crit_n],
        hole=0.7, sort=False,
        marker=dict(colors=["#34d399","#fbbf24","#f87171"], line=dict(color="#06090f", width=3)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,} drives<extra></extra>",
    ))
    fig_d.update_layout(
        **PLT, height=280, showlegend=False,
        annotations=[dict(
            text=f"<b style='font-size:32px;color:#eaeef3;font-family:JetBrains Mono,monospace'>{avg_h:.0f}</b>"
                 f"<br><span style='font-size:11px;color:#4a5568'>HEALTH</span>",
            x=0.5, y=0.5, showarrow=False, font_size=14,
        )]
    )
    st.plotly_chart(fig_d, use_container_width=True, key="donut")

with col2:
    # Health distribution — clean area chart
    bins = np.arange(0, 105, 2.5)
    hist_vals, bin_edges = np.histogram(fdf["health_score"], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=bin_centers, y=hist_vals, mode="lines",
        fill="tozeroy", fillcolor="rgba(226,35,26,0.08)",
        line=dict(color="#E2231A", width=2, shape="spline"),
        hovertemplate="Score: %{x:.0f}<br>Drives: %{y}<extra></extra>",
    ))
    fig_area.add_vline(x=avg_h, line_dash="dot", line_color="#34d399", line_width=1,
                       annotation_text=f"Mean {avg_h:.1f}", annotation_font_color="#34d399",
                       annotation_font_size=10)
    fig_area.update_layout(**PLT, height=280, xaxis_title=None, yaxis_title=None,
                           xaxis=dict(range=[45, 100], gridcolor="rgba(255,255,255,0.02)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.02)"))
    st.plotly_chart(fig_area, use_container_width=True, key="healtharea")

with col3:
    # Failure by vendor — horizontal lollipop
    vs = fdf.groupby("Vendor").agg(tot=("Failure_Flag","count"), fail=("Failure_Flag","sum")).reset_index()
    vs["rate"] = vs["fail"] / vs["tot"] * 100
    vs = vs.sort_values("rate", ascending=True)
    fig_lol = go.Figure()
    for _, row in vs.iterrows():
        color = "#f87171" if row["rate"] > 2 else "#fbbf24" if row["rate"] > 1 else "#34d399"
        fig_lol.add_trace(go.Scatter(
            x=[0, row["rate"]], y=[row["Vendor"], row["Vendor"]],
            mode="lines", line=dict(color=color, width=3), showlegend=False,
            hoverinfo="skip",
        ))
        fig_lol.add_trace(go.Scatter(
            x=[row["rate"]], y=[row["Vendor"]],
            mode="markers+text", marker=dict(size=10, color=color),
            text=[f"  {row['rate']:.1f}%"], textposition="middle right",
            textfont=dict(size=11, color="#eaeef3", family="JetBrains Mono, monospace"),
            showlegend=False,
            hovertemplate=f"<b>{row['Vendor']}</b><br>{row['rate']:.1f}% failure rate<br>{int(row['fail'])} of {int(row['tot'])} drives<extra></extra>",
        ))
    fig_lol.update_layout(**PLT, height=280, xaxis_title=None, yaxis_title=None,
                          xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                          yaxis=dict(showgrid=False))
    st.plotly_chart(fig_lol, use_container_width=True, key="vendorlol")


# ━━━━━━━━━━━━━━━ MAIN TABS ━━━━━━━━━━━━━━━
st.markdown('<div class="sec-head"><h2>Analytics</h2><span class="tag">Interactive</span></div>', unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["Failure Patterns", "Feature Intelligence", "Drive Explorer", "Model"])

# ─── TAB 1: Failure Patterns ───
with t1:
    c_left, c_right = st.columns([3, 2])
    with c_left:
        mode_counts = fdf["Failure_Mode_Label"].value_counts().reset_index()
        mode_counts.columns = ["Mode", "Count"]
        fig_bar = go.Figure()
        for _, row in mode_counts.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[row["Count"]], y=[row["Mode"]], orientation="h",
                marker_color=MODE_C.get(row["Mode"], "#60a5fa"),
                marker_line_width=0, opacity=0.88, showlegend=False,
                text=[f"  {row['Count']:,}"], textposition="outside",
                textfont=dict(color="#eaeef3", size=12, family="JetBrains Mono, monospace"),
                hovertemplate=f"<b>{row['Mode']}</b><br>{row['Count']:,} drives<extra></extra>",
            ))
        fig_bar.update_layout(**PLT, height=340, xaxis=dict(showgrid=False, showticklabels=False),
                              yaxis=dict(showgrid=False), barmode="stack")
        st.plotly_chart(fig_bar, use_container_width=True, key="failbar")

    with c_right:
        # Failure mode radar
        radar_features = ["Temperature_C", "Media_Errors", "Percent_Life_Used",
                          "Unsafe_Shutdowns", "CRC_Errors", "Power_On_Hours"]
        mode_means = fdf.groupby("Failure_Mode_Label")[radar_features].mean()
        mm = (mode_means - mode_means.min()) / (mode_means.max() - mode_means.min() + 1e-9)
        fig_r = go.Figure()
        for name in mm.index:
            if name == "None":
                continue
            vals = mm.loc[name].tolist() + [mm.loc[name].tolist()[0]]
            short_labels = ["Temp", "Media Err", "Life Used", "Unsafe Shut", "CRC Err", "POH"]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=short_labels + [short_labels[0]],
                name=name, fill="toself",
                line=dict(color=MODE_C.get(name, "#60a5fa"), width=2),
                fillcolor=MODE_C.get(name, "#60a5fa"), opacity=0.12,
            ))
        fig_r.update_layout(
            **PLT, height=340, showlegend=True,
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                        font=dict(size=10, color="#7a8599")),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1.05], gridcolor="rgba(255,255,255,0.03)",
                                tickfont=dict(size=8, color="#4a5568")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.03)",
                                 tickfont=dict(size=10, color="#7a8599")),
            ),
        )
        st.plotly_chart(fig_r, use_container_width=True, key="radar")


# ─── TAB 2: Feature Intelligence ───
with t2:
    c_fi, c_bx = st.columns(2)

    with c_fi:
        imp = importance_df.sort_values("importance", ascending=True).tail(10)
        max_imp = imp["importance"].max()
        bars_html = ""
        for _, row in imp.iterrows():
            pct = row["importance"] / max_imp * 100
            bars_html += f"""
            <div class="stat-bar">
              <div class="name">{row['feature']}</div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.0f}%; background: linear-gradient(90deg, var(--accent), #fb923c);"></div>
              </div>
              <div class="val">{row['importance']:.4f}</div>
            </div>"""
        st.markdown(f'<div class="panel"><div class="panel-title">Feature Importance</div>{bars_html}</div>', unsafe_allow_html=True)

    with c_bx:
        feat = st.selectbox("Compare feature by class", radar_features, index=0, key="feat_sel")
        fig_v = px.violin(
            fdf, y=feat, x="Failure_Flag", color="Failure_Flag",
            color_discrete_map={0: "#34d399", 1: "#f87171"},
            box=True, points=False,
            labels={"Failure_Flag": ""},
        )
        fig_v.update_layout(**PLT, height=400, showlegend=False,
                            xaxis=dict(ticktext=["Healthy","Failed"], tickvals=[0,1]))
        st.plotly_chart(fig_v, use_container_width=True, key="violin")

    # Correlation heatmap (full width)
    numeric_cols = fdf.select_dtypes(include=[np.number]).columns.tolist()
    exc = ["Failure_Mode", "Failure_Flag", "health_score", "failure_probability"]
    cc = [c for c in numeric_cols if c not in exc]
    corr = fdf[cc].corr()
    fig_hm = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0,"#1e3a5f"],[0.5,"#06090f"],[1,"#E2231A"]],
        zmid=0, text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
        textfont=dict(size=9, color="#7a8599"),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        showscale=False
    ))
    fig_hm.update_layout(**PLT, height=420, xaxis_tickangle=-40,
                         margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_hm, use_container_width=True, key="corr")


# ─── TAB 3: Drive Explorer ───
with t3:
    search = st.text_input("Search by Drive ID", "", placeholder="e.g. DRV-0042", key="search_drv")

    edf = fdf[["Drive_ID","Vendor","Model","health_score","status","failure_probability",
               "Failure_Mode_Label","Temperature_C","Media_Errors","Percent_Life_Used",
               "CRC_Errors","reasons","recommendations"]].copy()

    if search:
        edf = edf[edf["Drive_ID"].str.contains(search, case=False, na=False)]

    sort_by = st.selectbox("Sort", ["health_score","failure_probability","Temperature_C","Media_Errors"], key="sort_sel")
    edf = edf.sort_values(sort_by, ascending=True)

    st.markdown(f"**{len(edf):,} drives**")
    st.dataframe(
        edf.head(300).rename(columns={
            "Drive_ID":"Drive","health_score":"Health","status":"Status",
            "failure_probability":"Fail %","Failure_Mode_Label":"Mode",
            "Temperature_C":"Temp","Media_Errors":"Media","Percent_Life_Used":"Life %",
            "CRC_Errors":"CRC"
        })[["Drive","Vendor","Model","Health","Status","Fail %","Mode","Temp","Media","Life %","CRC"]],
        use_container_width=True, height=420
    )

    if len(edf) > 0:
        sel = st.selectbox("Select drive for detail", edf["Drive_ID"].head(50).tolist(), key="drv_detail")
        if sel:
            drv = edf[edf["Drive_ID"]==sel].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Health", f"{drv['health_score']:.1f}")
            m2.metric("Status", drv["status"])
            m3.metric("Fail Prob", f"{drv['failure_probability']:.1f}%")
            m4.metric("Mode", drv["Failure_Mode_Label"])
            r1, r2 = st.columns(2)
            with r1:
                st.caption("RISK FACTORS")
                st.info(drv["reasons"])
            with r2:
                st.caption("RECOMMENDATIONS")
                st.warning(drv["recommendations"])


# ─── TAB 4: Model ───
with t4:
    # Metrics bar
    met_html = ""
    for label, val, color in [("Accuracy",metrics["accuracy"],"--green"),
                               ("Precision",metrics["precision"],"--blue"),
                               ("Recall",metrics["recall"],"--yellow"),
                               ("F1",metrics["f1_score"],"--purple")]:
        pct = val
        met_html += f"""
        <div class="stat-bar">
          <div class="name">{label}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.0f}%; background: var({color});"></div>
          </div>
          <div class="val">{val:.1f}%</div>
        </div>"""
    st.markdown(f'<div class="panel"><div class="panel-title">Classification Metrics</div>{met_html}</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        cm = metrics["confusion_matrix"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred Healthy","Pred Failed"], y=["True Healthy","True Failed"],
            colorscale=[[0,"#06090f"],[1,"#E2231A"]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont=dict(size=22, color="#eaeef3"), showscale=False,
            hovertemplate="%{y} -> %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(**PLT, height=340, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_cm, use_container_width=True, key="cm")

    with mc2:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">Summary</div>
          <div class="stat-bar"><div class="name">True Negatives</div><div class="val c-green">{tn:,}</div></div>
          <div class="stat-bar"><div class="name">True Positives</div><div class="val c-green">{tp:,}</div></div>
          <div class="stat-bar"><div class="name">False Positives</div><div class="val c-yellow">{fp:,}</div></div>
          <div class="stat-bar"><div class="name">False Negatives</div><div class="val c-red">{fn:,}</div></div>
        </div>
        """, unsafe_allow_html=True)

        if fn == 0:
            st.success("Zero missed failures - every failing drive is detected before failure.")
        else:
            st.warning(f"{fn} failure(s) missed - consider tuning recall threshold.")

    # Feature importance detailed
    imp_s = importance_df.sort_values("importance", ascending=False)
    fig_fi = go.Figure(go.Bar(
        y=imp_s["feature"], x=imp_s["importance"], orientation="h",
        marker=dict(color=imp_s["importance"],
                    colorscale=[[0,"#1e3a5f"],[0.5,"#E2231A"],[1,"#fb923c"]]),
        text=imp_s["importance"].apply(lambda v: f"{v:.4f}"),
        textposition="outside", textfont=dict(color="#7a8599", size=11),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(**PLT, height=max(280, len(imp_s)*28),
                         yaxis=dict(autorange="reversed", showgrid=False),
                         xaxis=dict(showgrid=False, showticklabels=False))
    st.plotly_chart(fig_fi, use_container_width=True, key="fi_detail")


# ━━━━━━━━━━━━━━━ FOOTER ━━━━━━━━━━━━━━━
st.markdown(f"""
<div class="foot">
  Powered by <b>NVMe Health Intelligence Engine</b> &middot; Built for <b>Lenovo</b> &middot;
  {total:,} drives &middot; {datetime.now().strftime('%d %b %Y')}
</div>
""", unsafe_allow_html=True)
