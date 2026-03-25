"""
Data Processing Layer
======================
Handles missing values, feature scaling, and feature selection
for the NVMe telemetry dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Columns to exclude from ML features
EXCLUDE_COLS = [
    "Drive_ID", "Vendor", "Model", "Firmware_Version",
    "Failure_Flag", "Failure_Mode", "Failure_Mode_Label",
]


def load_and_clean(filepath="data/nvme_dataset.csv"):
    """Load the dataset and handle missing values."""
    from data_generator import load_dataset
    df = load_dataset(filepath)

    # Report missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"[i] Missing values found:")
        for col, cnt in missing.items():
            print(f"    {col}: {cnt} missing ({cnt/len(df)*100:.1f}%)")

        # Impute missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    → {col}: filled with median = {median_val}")
    else:
        print("[✓] No missing values found")

    print(f"[✓] Data cleaning complete. Remaining nulls: {df.isnull().sum().sum()}")
    return df


def prepare_features(df):
    """
    Prepare features and target variable.
    Returns: X (features DataFrame), y (target Series), feature_names (list)
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df["Failure_Flag"].copy()

    print(f"[✓] Features prepared: {len(feature_cols)} features")
    print(f"    Features: {feature_cols}")
    print(f"    Target distribution: Healthy={int((y==0).sum())}, Failed={int((y==1).sum())}")
    return X, y, feature_cols


def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler.
    Returns: X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    print(f"[✓] Feature scaling complete (StandardScaler)")
    return X_train_scaled, X_test_scaled, scaler


def get_summary_statistics(df):
    """Return a summary statistics table for EDA."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats["missing"] = df.isnull().sum()
    return stats


if __name__ == "__main__":
    df = load_and_clean()
    X, y, features = prepare_features(df)
    print("\nSummary Statistics:")
    print(get_summary_statistics(df).to_string())
