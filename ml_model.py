"""
Machine Learning Prediction Layer
===================================
Trains a Random Forest classifier on NVMe telemetry data,
evaluates performance, and provides failure probability predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[✓] Data split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"    Train failure rate: {y_train.mean()*100:.1f}%")
    print(f"    Test  failure rate: {y_test.mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[✓] Random Forest model trained (200 trees, max_depth=12)")
    return model


def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier (baseline)."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[✓] Logistic Regression model trained (baseline)")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall   : {metrics['recall']:.2f}%")
    print(f"  F1 Score : {metrics['f1_score']:.2f}%")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
    print(f"{'='*50}")

    return metrics, y_pred, y_proba


def get_feature_importance(model, feature_names):
    """Extract and rank feature importances from tree-based model."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n[✓] Feature Importance Ranking:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"    {row['feature']:25s} {row['importance']:.4f} {bar}")

    return importance_df


def predict_failure_probability(model, X):
    """Return failure probability (0–100%) for each drive."""
    proba = model.predict_proba(X)[:, 1] * 100
    return np.round(proba, 2)


if __name__ == "__main__":
    from data_processing import load_and_clean, prepare_features, scale_features

    df = load_and_clean()
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    rf_model = train_random_forest(X_train_s, y_train)
    evaluate_model(rf_model, X_test_s, y_test, "Random Forest")
    get_feature_importance(rf_model, feature_names)
