"""
Intelligence Layer
===================
Three engines:
  1. Health Score Engine   — computes 0–100 health score per drive
  2. Explainability Engine — identifies top reasons for failure risk
  3. Recommendation Engine — maps risk factors to actionable advice
"""

import pandas as pd
import numpy as np


# ──────────────── Weights for Health Score ────────────────
HEALTH_WEIGHTS = {
    "Temperature_C":      0.20,
    "Media_Errors":       0.25,
    "CRC_Errors":         0.10,
    "Percent_Life_Used":  0.25,
    "Unsafe_Shutdowns":   0.10,
    "SMART_Warning_Flag": 0.10,   # 1 = warning present → high penalty
}

# Normalization ranges (expected min/max for each feature)
FEATURE_RANGES = {
    "Temperature_C":      (20, 85),
    "Media_Errors":       (0, 60),
    "CRC_Errors":         (0, 50),
    "Percent_Life_Used":  (0, 100),
    "Unsafe_Shutdowns":   (0, 60),
    "SMART_Warning_Flag": (0, 1),
}

# ──────────────── Thresholds for risk flags ────────────────
RISK_THRESHOLDS = {
    "Temperature_C": {"warn": 50, "critical": 65,
        "reason": "High operating temperature ({val}°C)",
        "recommendation": "Improve cooling — ensure adequate airflow and check thermal paste"},
    "Media_Errors": {"warn": 5, "critical": 15,
        "reason": "Elevated media error count ({val})",
        "recommendation": "Back up data immediately — drive media degradation detected"},
    "CRC_Errors": {"warn": 5, "critical": 12,
        "reason": "High CRC error count ({val})",
        "recommendation": "Check cable/connector integrity and power supply stability"},
    "Percent_Life_Used": {"warn": 60, "critical": 80,
        "reason": "High wear level ({val}%)",
        "recommendation": "Plan drive replacement — endurance nearing limit"},
    "Unsafe_Shutdowns": {"warn": 10, "critical": 25,
        "reason": "Frequent unsafe shutdowns ({val})",
        "recommendation": "Stabilize power supply — use UPS and check power connections"},
    "SMART_Warning_Flag": {"warn": 1, "critical": 1,
        "reason": "SMART warning flag active",
        "recommendation": "Investigate SMART health status — potential imminent failure"},
}


def _normalize(value, vmin, vmax):
    """Normalize value to [0, 1] range."""
    return np.clip((value - vmin) / (vmax - vmin + 1e-9), 0, 1)


# ═══════════════════════════════════════════════════════════
#  1. HEALTH SCORE ENGINE
# ═══════════════════════════════════════════════════════════
def compute_health_score(row):
    """
    Compute health score (0–100) for a single drive.
    Health Score = 100 - weighted sum of normalized risk factors
    """
    penalty = 0.0
    for feature, weight in HEALTH_WEIGHTS.items():
        val = row.get(feature, 0)
        vmin, vmax = FEATURE_RANGES[feature]
        norm = _normalize(val, vmin, vmax)
        penalty += weight * norm

    score = 100 * (1 - penalty)
    return round(max(0, min(100, score)), 1)


def classify_health(score):
    """Classify health score into status category."""
    if score >= 80:
        return "Healthy"
    elif score >= 50:
        return "Warning"
    else:
        return "Critical"


# ═══════════════════════════════════════════════════════════
#  2. EXPLAINABILITY ENGINE
# ═══════════════════════════════════════════════════════════
def explain_risk(row):
    """
    Identify the key reasons why a drive is at risk.
    Returns a list of (severity, reason) tuples.
    """
    reasons = []

    for feature, thresholds in RISK_THRESHOLDS.items():
        val = row.get(feature, 0)

        if val >= thresholds.get("critical", float("inf")):
            reasons.append(("CRITICAL", thresholds["reason"].format(val=val)))
        elif val >= thresholds.get("warn", float("inf")):
            reasons.append(("WARNING", thresholds["reason"].format(val=val)))

    if not reasons:
        reasons.append(("OK", "All parameters within normal range"))

    return reasons


# ═══════════════════════════════════════════════════════════
#  3. RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════
def generate_recommendations(row):
    """
    Generate actionable recommendations based on drive telemetry.
    Returns a list of recommendation strings.
    """
    recommendations = []

    for feature, thresholds in RISK_THRESHOLDS.items():
        val = row.get(feature, 0)

        if val >= thresholds.get("critical", float("inf")):
            recommendations.append(f"🔴 {thresholds['recommendation']}")
        elif val >= thresholds.get("warn", float("inf")):
            recommendations.append(f"🟡 {thresholds['recommendation']}")

    if not recommendations:
        recommendations.append("✅ No action required — drive is in good health")

    return recommendations


# ═══════════════════════════════════════════════════════════
#  BATCH PROCESSING
# ═══════════════════════════════════════════════════════════
def analyze_drive(row, failure_probability=None):
    """Full analysis for a single drive. Returns a result dict."""
    score = compute_health_score(row)
    status = classify_health(score)
    reasons = explain_risk(row)
    recs = generate_recommendations(row)

    # Build human-readable reason string
    reason_str = "; ".join([f"[{lvl}] {msg}" for lvl, msg in reasons])

    return {
        "drive_id": row.get("Drive_ID", "Unknown"),
        "health_score": score,
        "status": status,
        "failure_probability": failure_probability,
        "reasons": reason_str,
        "recommendations": " | ".join(recs),
        "reason_list": reasons,
        "recommendation_list": recs,
    }


def analyze_all_drives(df, failure_probabilities=None):
    """Analyze all drives and return results DataFrame."""
    results = []
    for idx, row in df.iterrows():
        fp = failure_probabilities[idx] if failure_probabilities is not None else None
        results.append(analyze_drive(row, fp))

    results_df = pd.DataFrame(results)
    print(f"\n[✓] Intelligence analysis complete for {len(results_df)} drives")

    # Summary
    status_counts = results_df["status"].value_counts()
    print("    Status Distribution:")
    for status, count in status_counts.items():
        emoji = {"Healthy": "🟢", "Warning": "🟡", "Critical": "🔴"}.get(status, "⚪")
        print(f"      {emoji} {status}: {count} drives ({count/len(results_df)*100:.1f}%)")

    avg_score = results_df["health_score"].mean()
    print(f"    Average Health Score: {avg_score:.1f}/100")

    return results_df


if __name__ == "__main__":
    from data_processing import load_and_clean
    df = load_and_clean()
    results = analyze_all_drives(df)
    print("\nSample Results (top 10):")
    print(results[["drive_id", "health_score", "status", "reasons"]].head(10).to_string(index=False))
