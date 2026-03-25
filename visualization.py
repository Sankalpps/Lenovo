"""
Visualization Layer
====================
Generates publication-quality charts for the NVMe Health Intelligence system.
All plots are saved to the output/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────── Style Configuration ────────────────
plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#161B22",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#C9D1D9",
    "text.color": "#C9D1D9",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#21262D",
    "font.family": "sans-serif",
    "font.size": 11,
})

OUTPUT_DIR = "output"

# Color palettes
PALETTE_MAIN = ["#58A6FF", "#F78166", "#7EE787", "#D2A8FF", "#FF7B72", "#79C0FF"]
PALETTE_HEAT = "RdYlGn_r"
PALETTE_STATUS = {"Healthy": "#7EE787", "Warning": "#F0C75E", "Critical": "#FF7B72"}
PALETTE_MODES = {
    "None": "#58A6FF", "Wear-Out": "#FF7B72", "Thermal": "#F78166",
    "Power": "#D2A8FF", "Firmware": "#79C0FF", "Early-Life": "#F0C75E",
}


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    📊 Saved: {path}")


# ═══════════════════ 1. Failure Distribution ═══════════════════
def plot_failure_distribution(df):
    """Bar chart of failure mode distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df["Failure_Mode_Label"].value_counts()
    colors = [PALETTE_MODES.get(m, "#58A6FF") for m in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="#30363D", linewidth=0.8)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=12, color="#C9D1D9")

    ax.set_title("NVMe Drive Failure Mode Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Failure Mode", fontsize=12)
    ax.set_ylabel("Number of Drives", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "01_failure_distribution.png")


# ═══════════════════ 2. Feature Importance ═══════════════════
def plot_feature_importance(importance_df):
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    imp = importance_df.sort_values("importance", ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(imp)))
    ax.barh(imp["feature"], imp["importance"], color=colors, edgecolor="#30363D", height=0.6)

    for i, (val, name) in enumerate(zip(imp["importance"], imp["feature"])):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=10, color="#C9D1D9")

    ax.set_title("Feature Importance (Random Forest)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Importance", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "02_feature_importance.png")


# ═══════════════════ 3. Correlation Heatmap ═══════════════════
def plot_correlation_heatmap(corr_matrix):
    """Heatmap of feature correlations."""
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, ax=ax, linewidths=0.5, linecolor="#30363D",
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    _save(fig, "03_correlation_heatmap.png")


# ═══════════════════ 4. Health Score Distribution ═══════════════════
def plot_health_score_distribution(results_df):
    """Histogram of health scores with status zones."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Background zones
    ax.axvspan(0, 49, alpha=0.12, color="#FF7B72", label="Critical (0–49)")
    ax.axvspan(50, 79, alpha=0.12, color="#F0C75E", label="Warning (50–79)")
    ax.axvspan(80, 100, alpha=0.12, color="#7EE787", label="Healthy (80–100)")

    # Histogram
    scores = results_df["health_score"]
    ax.hist(scores, bins=30, color="#58A6FF", edgecolor="#30363D", alpha=0.85, linewidth=0.8)

    # Mean line
    mean_score = scores.mean()
    ax.axvline(mean_score, color="#F78166", linestyle="--", linewidth=2,
               label=f"Mean = {mean_score:.1f}")

    ax.set_title("Drive Health Score Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Health Score", fontsize=12)
    ax.set_ylabel("Number of Drives", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.7, fontsize=10)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "04_health_score_distribution.png")


# ═══════════════════ 5. Confusion Matrix ═══════════════════
def plot_confusion_matrix(cm):
    """Heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ["Healthy", "Failed"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=1, linecolor="#30363D",
        annot_kws={"size": 18, "weight": "bold"},
    )
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    _save(fig, "05_confusion_matrix.png")


# ═══════════════════ 6. Feature Boxplots ═══════════════════
def plot_feature_boxplots(df):
    """Box-plots of key features grouped by failure status."""
    features = ["Temperature_C", "Media_Errors", "Percent_Life_Used",
                "Unsafe_Shutdowns", "CRC_Errors", "SMART_Warning_Flag"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for ax, feat in zip(axes.flatten(), features):
        data_healthy = df[df["Failure_Flag"] == 0][feat].dropna()
        data_failed  = df[df["Failure_Flag"] == 1][feat].dropna()

        bp = ax.boxplot(
            [data_healthy, data_failed],
            labels=["Healthy", "Failed"],
            patch_artist=True,
            widths=0.5,
            boxprops=dict(linewidth=1.2),
            medianprops=dict(color="#F78166", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#7EE787")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#FF7B72")
        bp["boxes"][1].set_alpha(0.6)

        ax.set_title(feat.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Feature Distribution: Healthy vs Failed Drives",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "06_feature_boxplots.png")


# ═══════════════════ 7. Status Pie Chart ═══════════════════
def plot_status_pie(results_df):
    """Pie chart of health status distribution."""
    fig, ax = plt.subplots(figsize=(8, 8))
    status_counts = results_df["status"].value_counts()
    colors = [PALETTE_STATUS.get(s, "#58A6FF") for s in status_counts.index]
    wedges, texts, autotexts = ax.pie(
        status_counts.values, labels=status_counts.index, colors=colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.82,
        wedgeprops=dict(width=0.4, edgecolor="#0D1117", linewidth=2),
        textprops=dict(fontsize=13, fontweight="bold"),
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
        autotext.set_color("#0D1117")

    ax.set_title("Drive Health Status Overview", fontsize=16, fontweight="bold", pad=20)
    _save(fig, "07_status_pie_chart.png")


# ═══════════════════ Run All ═══════════════════
def generate_all_visualizations(df, results_df, corr_matrix, importance_df, confusion_mat):
    """Generate all visualizations."""
    print("\n[✓] Generating visualizations...")
    plot_failure_distribution(df)
    plot_feature_importance(importance_df)
    plot_correlation_heatmap(corr_matrix)
    plot_health_score_distribution(results_df)
    plot_confusion_matrix(confusion_mat)
    plot_feature_boxplots(df)
    plot_status_pie(results_df)
    print(f"[✓] All visualizations saved to {OUTPUT_DIR}/")
