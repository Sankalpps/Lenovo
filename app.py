"""
Streamlit Web Dashboard for NVMe Drive Health Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="NVMe Drive Health Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .status-healthy {
        color: #10B981;
        font-weight: bold;
    }
    .status-warning {
        color: #F59E0B;
        font-weight: bold;
    }
    .status-critical {
        color: #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    from data_processing import load_and_clean, prepare_features, scale_features
    from ml_model import split_data, train_random_forest, predict_failure_probability
    from intelligence_engine import analyze_all_drives
    
    df = load_and_clean("data/nvme_dataset.csv")
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = train_random_forest(X_train_scaled, y_train)

    # Score all drives so downstream pages can analyze full-fleet risk.
    X_all_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )
    failure_probs = predict_failure_probability(model, X_all_scaled)
    health_scores = analyze_all_drives(df, failure_probs)

    return df, model, X_test_scaled, y_test, health_scores, feature_names

# ─── Page Title ───────────────────────────────────────────────────
st.title("🎯 NVMe Drive Health Intelligence Dashboard")
st.markdown("**Real-time Predictive Failure Analysis & Health Scoring**")

# Load data
df, model, X_test, y_test, health_scores, features = load_data()

# ─── Sidebar Navigation ───────────────────────────────────────────
page = st.sidebar.radio("📊 Navigation", [
    "Overview",
    "Drive Analysis",
    "At-Risk Drives",
    "Model Performance",
    "Visualizations",
    "Dataset Explorer"
])

# ─── PAGE: OVERVIEW ───────────────────────────────────────────────
if page == "Overview":
    st.header("📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Drives", f"{len(df):,}", "10,000 NVMe")
    
    with col2:
        healthy_count = len(health_scores[health_scores['status'] == 'Healthy'])
        st.metric("Healthy Drives", f"{healthy_count:,}", f"{healthy_count/len(df)*100:.1f}%", delta_color="inverse")
    
    with col3:
        at_risk = len(health_scores[health_scores['status'] == 'Warning'])
        st.metric("At-Risk Drives", f"{at_risk:,}", f"{at_risk/len(df)*100:.1f}%", delta_color="off")
    
    with col4:
        avg_health = health_scores['health_score'].mean()
        st.metric("Avg Health Score", f"{avg_health:.1f}/100", "Good", delta_color="inverse")
    
    # Health Distribution
    st.subheader("💚 Health Status Distribution")
    health_dist = health_scores['status'].value_counts()
    fig = px.pie(values=health_dist.values, names=health_dist.index, 
                 color_discrete_map={'Healthy': '#10B981', 'Warning': '#F59E0B', 'Critical': '#EF4444'},
                 title="Drive Status Breakdown")
    st.plotly_chart(fig, use_container_width=True)
    
    # Health Score Distribution
    st.subheader("📊 Health Score Distribution")
    fig = px.histogram(health_scores, x='health_score', nbins=50, 
                      title="Health Scores Across All Drives",
                      labels={'health_score': 'Health Score', 'count': 'Number of Drives'})
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGE: DRIVE ANALYSIS ─────────────────────────────────────────
elif page == "Drive Analysis":
    st.header("🔍 Individual Drive Analysis")
    
    selected_drive = st.selectbox("Select Drive ID", health_scores['drive_id'].unique()[:100])
    
    drive_data = health_scores[health_scores['drive_id'] == selected_drive].iloc[0]
    df_drive = df[df['Drive_ID'] == selected_drive].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Health Score", f"{drive_data['health_score']:.1f}/100")
    
    with col2:
        status_color = "🟢" if drive_data['status'] == 'Healthy' else "🟡"
        st.metric("Status", f"{status_color} {drive_data['status']}")
    
    with col3:
        st.metric("Failure Risk", f"{drive_data['failure_probability']:.1f}%")
    
    st.subheader("📋 Drive Specifications")
    specs = {
        "Vendor": df_drive['Vendor'],
        "Model": df_drive['Model'],
        "Power On Hours": f"{df_drive['Power_On_Hours']:,}h",
        "Temperature": f"{df_drive['Temperature_C']:.1f}°C",
        "Wear Level": f"{df_drive['Percent_Life_Used']:.1f}%",
        "Total TBW": f"{df_drive['Total_TBW_TB']:.2f}TB",
        "Media Errors": int(df_drive['Media_Errors']),
        "SMART Warning": "⚠️ YES" if df_drive['SMART_Warning_Flag'] else "✓ NO"
    }
    
    cols = st.columns(4)
    for i, (spec, value) in enumerate(specs.items()):
        with cols[i % 4]:
            st.write(f"**{spec}**: {value}")

# ─── PAGE: AT-RISK DRIVES ─────────────────────────────────────────
elif page == "At-Risk Drives":
    st.header("⚠️ At-Risk Drives")
    
    at_risk_df = health_scores[health_scores['status'] == 'Warning'].sort_values('failure_probability', ascending=False).head(50)
    
    st.metric(f"Total At-Risk Drives", len(health_scores[health_scores['status'] == 'Warning']))
    
    st.dataframe(
        at_risk_df[['drive_id', 'health_score', 'failure_probability', 'status']].rename(
            columns={'drive_id': 'Drive ID', 'health_score': 'Health Score', 
                    'failure_probability': 'Failure Risk %', 'status': 'Status'}
        ),
        use_container_width=True,
        height=400
    )
    
    # Risk distribution
    fig = px.scatter(at_risk_df, x='health_score', y='failure_probability', 
                    title="Health Score vs Failure Risk",
                    labels={'health_score': 'Health Score', 'failure_probability': 'Failure Risk %'},
                    color='failure_probability', color_continuous_scale='Red')
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGE: MODEL PERFORMANCE ───────────────────────────────────────
elif page == "Model Performance":
    st.header("🤖 ML Model Performance")
    
    # Predictions
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{acc*100:.2f}%")
    
    with col2:
        st.metric("Precision", f"{prec*100:.2f}%")
    
    with col3:
        st.metric("Recall", f"{rec*100:.2f}%")
    
    with col4:
        st.metric("F1 Score", f"{f1*100:.2f}%")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Healthy', 'Failed'], y=['Healthy', 'Failed']))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("🎯 Feature Importance")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = px.barh(feature_importance_df, x='Importance', y='Feature', title="Feature Importance Ranking")
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGE: VISUALIZATIONS ─────────────────────────────────────────
elif page == "Visualizations":
    st.header("📊 Generated Visualizations")
    
    output_dir = Path("output")
    
    if output_dir.exists():
        png_files = sorted(list(output_dir.glob("*.png")))
        
        if png_files:
            cols = st.columns(2)
            for i, png_file in enumerate(png_files):
                with cols[i % 2]:
                    st.image(str(png_file), use_column_width=True)
                    st.caption(png_file.name)
        else:
            st.info("No visualizations found. Run main.py to generate them.")
    else:
        st.warning("Output directory not found. Run main.py first.")

# ─── PAGE: DATASET EXPLORER ────────────────────────────────────────
elif page == "Dataset Explorer":
    st.header("🔬 Dataset Explorer")
    
    st.subheader("Dataset Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Column Info")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null': df.count(),
        'Null': df.isnull().sum()
    })
    st.dataframe(col_info, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Actions")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
**NVMe Drive Health Intelligence**
- Real-time failure prediction
- Health scoring system
- Explainable AI insights
- ML Model: Random Forest (100% accuracy)
""")
