"""
NVMe Drive Health Intelligence & Predictive Failure Analysis
=============================================================
Main pipeline script that orchestrates the full workflow:
  1. Load data
  2. Clean & process
  3. Exploratory data analysis
  4. Train ML model
  5. Evaluate performance
  6. Intelligence engine (health scores, explainability, recommendations)
  7. Generate visualizations
  8. Print console report
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

# Ensure src/ is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

warnings.filterwarnings("ignore")


def main():
    print("=" * 65)
    print("  NVMe Drive Health Intelligence & Predictive Failure Analysis")
    print("=" * 65)

    # ─── 1. Load & Clean Data ───────────────────────────────────
    print("\n>> STEP 1: Loading & Cleaning Data")
    print("-" * 45)
    from data_processing import load_and_clean, prepare_features, scale_features
    df = load_and_clean("data/nvme_dataset.csv")

    # ─── 2. Exploratory Data Analysis ───────────────────────────
    print("\n>> STEP 2: Exploratory Data Analysis")
    print("-" * 45)
    from eda_analysis import run_full_eda
    eda_results = run_full_eda(df)
    corr_matrix = eda_results["correlation_matrix"]

    # ─── 3. Prepare Features & Train/Test Split ─────────────────
    print("\n>> STEP 3: Feature Engineering & Split")
    print("-" * 45)
    X, y, feature_names = prepare_features(df)

    from ml_model import split_data, train_random_forest, evaluate_model
    from ml_model import get_feature_importance, predict_failure_probability
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ─── 4. Scale Features ─────────────────────────────────────
    print("\n>> STEP 4: Scaling Features")
    print("-" * 45)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # ─── 5. Train Random Forest Model ──────────────────────────
    print("\n>> STEP 5: Training ML Model")
    print("-" * 45)
    rf_model = train_random_forest(X_train_s, y_train)

    # ─── 6. Evaluate Model ─────────────────────────────────────
    print("\n>> STEP 6: Model Evaluation")
    print("-" * 45)
    metrics, y_pred, y_proba = evaluate_model(rf_model, X_test_s, y_test, "Random Forest")
    importance_df = get_feature_importance(rf_model, feature_names)

    # ─── 7. Intelligence Engine ────────────────────────────────
    print("\n>> STEP 7: Intelligence Engine")
    print("-" * 45)
    from intelligence_engine import analyze_all_drives

    # Get failure probability for ALL drives (train + test recombined)
    X_all_scaled = pd.DataFrame(
        scaler.transform(X), columns=X.columns, index=X.index
    )
    failure_probs = predict_failure_probability(rf_model, X_all_scaled)
    intelligence_results = analyze_all_drives(df, failure_probs)

    # ─── 8. Generate Visualizations ────────────────────────────
    print("\n>> STEP 8: Generating Visualizations")
    print("-" * 45)
    from visualization import generate_all_visualizations
    generate_all_visualizations(
        df=df,
        results_df=intelligence_results,
        corr_matrix=corr_matrix,
        importance_df=importance_df,
        confusion_mat=metrics["confusion_matrix"],
    )

    # ─── 9. Console Report ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL REPORT — Sample Drive Analysis (Top 15)")
    print("=" * 65)

    # Show a mix of healthy, warning, and critical drives
    sample = intelligence_results.sort_values("health_score").head(15)
    report_cols = ["drive_id", "health_score", "status", "failure_probability", "reasons"]
    print(sample[report_cols].to_string(index=False))

    # Summary statistics
    print("\n" + "-" * 65)
    print("  SUMMARY")
    print("-" * 65)
    total = len(intelligence_results)
    for status in ["Healthy", "Warning", "Critical"]:
        cnt = (intelligence_results["status"] == status).sum()
        print(f"  {status:10s}: {cnt:5d} drives ({cnt/total*100:.1f}%)")

    avg_score = intelligence_results["health_score"].mean()
    print(f"\n  Average Health Score : {avg_score:.1f} / 100")
    print(f"  Model Accuracy      : {metrics['accuracy']:.2f}%")
    print(f"  Model F1 Score      : {metrics['f1_score']:.2f}%")
    print(f"  Total Drives        : {total}")

    print("\n" + "=" * 65)
    print("  ✅ Pipeline complete! Visualizations saved to output/")
    print("=" * 65)


if __name__ == "__main__":
    main()
