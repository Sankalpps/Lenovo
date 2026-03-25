"""
Data Loader
============
Loads the NVMe Drive Failure Dataset and adds human-readable failure mode labels.
"""

import pandas as pd
import os

# Mapping of numeric Failure_Mode codes to descriptive labels
FAILURE_MODE_MAP = {
    0: "None",
    1: "Wear-Out",
    2: "Thermal",
    3: "Power",
    4: "Firmware",
    5: "Early-Life",
}


def load_dataset(filepath="data/nvme_dataset.csv"):
    """Load the NVMe dataset and add a human-readable failure mode label."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    # Add human-readable failure mode label
    df["Failure_Mode_Label"] = df["Failure_Mode"].map(FAILURE_MODE_MAP).fillna("Unknown")

    print(f"[✓] Dataset loaded: {filepath}  ({len(df)} drives, {len(df.columns)} columns)")
    print(f"    Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(f"\nFailure distribution:")
    print(df["Failure_Mode_Label"].value_counts())
