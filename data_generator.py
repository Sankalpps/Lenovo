"""
Data Loader
============
Loads the NVMe Drive Failure Dataset and adds human-readable failure mode labels.
"""

import pandas as pd
import os
import numpy as np

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
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        # Streamlit Cloud may deploy without the CSV file; generate a compatible
        # synthetic dataset so the app remains runnable.
        rng = np.random.default_rng(42)
        n = 10000

        df = pd.DataFrame({
            "Drive_ID": [f"NVME-{i:05d}" for i in range(n)],
            "Vendor": rng.choice(["Lenovo", "Samsung", "WD", "Micron"], size=n, p=[0.4, 0.25, 0.2, 0.15]),
            "Model": rng.choice(["LNV-512", "LNV-1TB", "PRO-980", "SN850X", "M600"], size=n),
            "Firmware_Version": rng.choice(["1.0.3", "1.1.0", "2.0.1", "3.2.4"], size=n),
            "Power_On_Hours": rng.integers(100, 60000, size=n),
            "Total_TBW_TB": np.round(rng.uniform(1, 300, size=n), 2),
            "Total_TBR_TB": np.round(rng.uniform(1, 320, size=n), 2),
            "Temperature_C": np.round(rng.normal(40, 7, size=n).clip(20, 85), 1),
            "Percent_Life_Used": np.round(rng.normal(22, 18, size=n).clip(0, 130), 1),
            "Media_Errors": rng.poisson(1.0, size=n),
            "Unsafe_Shutdowns": rng.poisson(3.0, size=n),
            "CRC_Errors": rng.poisson(0.5, size=n),
            "Read_Error_Rate": np.round(rng.gamma(2.0, 4.0, size=n), 2),
            "Write_Error_Rate": np.round(rng.gamma(2.0, 3.5, size=n), 2),
            "SMART_Warning_Flag": rng.choice([0, 1], size=n, p=[0.98, 0.02]),
        })

        # Build a failure signal from risk factors to keep the ML workflow meaningful.
        risk_score = (
            0.03 * df["Percent_Life_Used"]
            + 0.9 * df["SMART_Warning_Flag"]
            + 0.18 * df["Media_Errors"]
            + 0.01 * (df["Temperature_C"] - 30).clip(lower=0)
        )
        fail_prob = 1 / (1 + np.exp(-(risk_score - 3.5)))
        df["Failure_Flag"] = (rng.random(n) < fail_prob * 0.35).astype(int)

        mode_probs = np.array([0.00, 0.18, 0.16, 0.22, 0.27, 0.17])
        mode_probs = mode_probs / mode_probs.sum()
        failed_count = int(df["Failure_Flag"].sum())
        failed_modes = rng.choice([1, 2, 3, 4, 5], size=failed_count, p=mode_probs[1:])
        df["Failure_Mode"] = 0
        if failed_count > 0:
            df.loc[df["Failure_Flag"] == 1, "Failure_Mode"] = failed_modes

        print(f"[i] Dataset not found at '{filepath}'. Generated synthetic dataset ({n} drives).")

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
