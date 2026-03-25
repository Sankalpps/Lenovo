# NVMe Drive Health Intelligence & Predictive Failure Analysis

A Python-based system that transforms NVMe SMART telemetry into actionable insights with failure prediction, health scoring, and explainable recommendations.

## Features

- **Data Processing** — Automated cleaning, imputation, and feature scaling
- **Exploratory Analysis** — Correlation heatmaps, failure distributions, pattern identification
- **ML Prediction** — Random Forest classifier with accuracy, precision, recall, F1 metrics
- **Health Scoring** — Weighted 0–100 health score per drive (Healthy / Warning / Critical)
- **Explainability** — Per-drive risk factor identification with severity levels
- **Recommendations** — Actionable maintenance advice based on telemetry patterns
- **Visualizations** — 7 publication-quality charts saved automatically

## Project Structure

```
├── data/
│   └── nvme_dataset.csv          # NVMe Drive Failure Dataset (10,000 drives)
├── src/
│   ├── data_generator.py         # Dataset loader with failure mode mapping
│   ├── data_processing.py        # Cleaning, scaling, feature engineering
│   ├── eda_analysis.py           # Exploratory data analysis
│   ├── ml_model.py               # Random Forest training & evaluation
│   ├── intelligence_engine.py    # Health score, explainability, recommendations
│   └── visualization.py          # All charts and graphs
├── output/                       # Generated plots (created automatically)
├── main.py                       # Main pipeline script
├── requirements.txt              # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

This runs the full pipeline: data loading → cleaning → EDA → model training → evaluation → intelligence analysis → visualization → console report.

## Dataset

The system uses the **NVMe Drive Failure Dataset** with 10,000 drives and 17 features including:

| Feature | Description |
|---------|-------------|
| Temperature_C | Operating temperature (°C) |
| Power_On_Hours | Total hours powered on |
| Total_TBW_TB | Total terabytes written |
| Percent_Life_Used | Drive endurance consumed (%) |
| Media_Errors | Count of media/data errors |
| CRC_Errors | Count of CRC errors |
| Unsafe_Shutdowns | Count of unsafe shutdown events |
| SMART_Warning_Flag | SMART health warning (0/1) |
| Failure_Flag | Drive failure status (0 = healthy, 1 = failed) |
| Failure_Mode | Failure type: None, Wear-Out, Thermal, Power, Firmware, Early-Life |

## Output

Running the pipeline generates 7 plots in `output/`:

1. `01_failure_distribution.png` — Failure mode bar chart
2. `02_feature_importance.png` — Feature importance ranking
3. `03_correlation_heatmap.png` — Feature correlation matrix
4. `04_health_score_distribution.png` — Health score histogram
5. `05_confusion_matrix.png` — ML model confusion matrix
6. `06_feature_boxplots.png` — Feature distributions by failure status
7. `07_status_pie_chart.png` — Health status overview
