"""
Exploratory Data Analysis (EDA) Layer
======================================
Generates insights: correlation matrix, failure distributions,
feature analysis by failure class, and summary reports.
"""

import pandas as pd
import numpy as np


def compute_correlation_matrix(df):
    """Compute Pearson correlation matrix for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    print("[✓] Correlation matrix computed")
    return corr


def failure_distribution(df):
    """Count drives per failure mode."""
    dist = df["Failure_Mode_Label"].value_counts()
    print("\n[✓] Failure Mode Distribution:")
    for mode, count in dist.items():
        pct = count / len(df) * 100
        print(f"    {mode:12s}: {count:5d} drives ({pct:.1f}%)")
    return dist


def feature_stats_by_class(df):
    """Compute mean of each numeric feature grouped by Failure_Flag."""
    exclude = ["Failure_Flag", "Failure_Mode"]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    grouped = df.groupby("Failure_Flag")[numeric_cols].mean()
    grouped.index = ["Healthy", "Failed"]
    print("\n[✓] Feature Means by Class:")
    print(grouped.T.to_string())
    return grouped


def identify_top_patterns(df):
    """
    Identify the top 5 failure patterns and their characteristics.
    Returns a dict of pattern -> description.
    """
    patterns = {}
    failed = df[df["Failure_Flag"] == 1]
    healthy = df[df["Failure_Flag"] == 0]

    comparisons = {
        "Temperature_C": ("High Temperature",
            "Drives with elevated operating temperatures show {ratio:.1f}x higher avg temp than healthy drives."),
        "Media_Errors": ("High Media Error Rate",
            "Failed drives exhibit {ratio:.1f}x more media errors on average."),
        "Percent_Life_Used": ("High Wear Level",
            "Failed drives have {ratio:.1f}x higher percentage used, indicating advanced wear."),
        "Unsafe_Shutdowns": ("Frequent Unsafe Shutdowns",
            "Failed drives experienced {ratio:.1f}x more unsafe shutdowns."),
        "CRC_Errors": ("CRC Error Accumulation",
            "Failed drives show {ratio:.1f}x higher CRC error counts."),
    }

    print("\n[✓] Top 5 Failure Patterns Identified:")
    for col, (name, template) in comparisons.items():
        h_mean = healthy[col].mean() if healthy[col].mean() != 0 else 0.01
        f_mean = failed[col].mean()
        ratio = f_mean / h_mean
        desc = template.format(ratio=ratio)
        patterns[name] = {
            "feature": col,
            "healthy_avg": round(h_mean, 2),
            "failed_avg": round(f_mean, 2),
            "ratio": round(ratio, 2),
            "description": desc,
        }
        print(f"    {name}: Failed avg={f_mean:.2f} vs Healthy avg={h_mean:.2f} (ratio={ratio:.1f}x)")

    return patterns


def run_full_eda(df):
    """Execute all EDA steps and return results dict."""
    results = {
        "correlation_matrix": compute_correlation_matrix(df),
        "failure_distribution": failure_distribution(df),
        "feature_stats": feature_stats_by_class(df),
        "top_patterns": identify_top_patterns(df),
    }
    return results


if __name__ == "__main__":
    from data_processing import load_and_clean
    df = load_and_clean()
    run_full_eda(df)
