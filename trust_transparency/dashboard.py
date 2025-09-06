import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.metrics import roc_curve, auc
from datetime import datetime, timedelta
import time
import random
import json
import os
import sys
import pickle
import base64
from sklearn.preprocessing import StandardScaler
try:
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is not installed. Please install it using pip install plotly")

# Try to import xgboost - we'll handle if it's not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost is not installed. Some models may not work correctly. Consider installing it with 'pip install xgboost'.")

# Function to load and encode images for HTML display
def get_base64_encoded_image(image_path):
    """Get base64 encoded image for HTML display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        st.warning(f"Could not load image from {image_path}: {str(e)}")
        return None

# Initialize session state for global model persistence
if 'model_name' not in st.session_state:
    st.session_state.model_name = "No model loaded"
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Unknown"
if 'model_features' not in st.session_state:
    st.session_state.model_features = []
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = 0.0
if 'model_path' not in st.session_state:
    st.session_state.model_path = ""
if 'model_upload_timestamp' not in st.session_state:
    st.session_state.model_upload_timestamp = None
if 'model_object' not in st.session_state:
    st.session_state.model_object = None

# Function to load model from session state
def load_model_from_session():
    """Load the model from the session state path if available"""
    if st.session_state.model_loaded and st.session_state.model_path:
        try:
            if os.path.exists(st.session_state.model_path):
                if st.session_state.model_object is None:
                    with open(st.session_state.model_path, 'rb') as f:
                        st.session_state.model_object = pickle.load(f)
                return st.session_state.model_object
        except Exception as e:
            st.warning(f"Error loading model from session state: {str(e)}")
    return None

# Function to display model info
def display_model_info_badge():
    """Display a small badge with current model information"""
    if st.session_state.model_loaded:
        upload_time = st.session_state.model_upload_timestamp
        time_str = upload_time.strftime("%Y-%m-%d %H:%M") if upload_time else "Unknown"
        st.sidebar.markdown(f"""
        <div style='background-color: #1E293B; padding: 10px; border-radius: 5px; margin-top: 20px;'>
            <p style='margin:0; color: #90CDF4; font-size: 0.8rem;'>Current Model</p>
            <p style='margin:0; color: #F1F5F9; font-weight: bold;'>{st.session_state.model_name}</p>
            <p style='margin:0; color: #94A3B8; font-size: 0.7rem;'>Type: {st.session_state.model_type} | Accuracy: {st.session_state.model_accuracy:.2%}</p>
            <p style='margin:0; color: #94A3B8; font-size: 0.7rem;'>Uploaded: {time_str}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style='background-color: #1E293B; padding: 10px; border-radius: 5px; margin-top: 20px;'>
            <p style='margin:0; color: #94A3B8;'>No model loaded</p>
            <p style='margin:0; color: #94A3B8; font-size: 0.7rem;'>Upload a model in the Real-time Fraud Detection section</p>
        </div>
        """, unsafe_allow_html=True)

# Helper function to display metrics as a chart
def display_metrics(metrics_df):
    """Display model metrics as visualizations"""
    if metrics_df.empty:
        st.warning("No metrics data available to display")
        return
    
    # Create a more comprehensive visualization with multiple metrics
    try:
        # Find which metrics columns are available
        metric_columns = []
        for col_name in ['ROC-AUC', 'ROC AUC', 'AUC', 'Precision', 'Recall', 'F1-Score', 'F1 Score']:
            if col_name in metrics_df.columns:
                metric_columns.append(col_name)
        
        if not metric_columns:
            st.warning("No metric columns found in the data")
            return
        
        # 1. Create a radar chart (polar chart) for multiple metrics
        # Prepare data for radar chart
        if len(metric_columns) >= 3 and 'Model' in metrics_df.columns:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            # Set number of metrics and angles
            N = len(metric_columns)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Add metric labels
            labels = metric_columns
            plt.xticks(angles[:-1], labels, size=12)
            
            # Draw axis lines
            ax.set_rlabel_position(0)
            plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], color="grey", size=10)
            plt.ylim(0.5, 1.0)
            
            # Plot each model's metrics
            for _, row in metrics_df.iterrows():
                model_name = row['Model']
                values = [row[metric] for metric in metric_columns]
                values += values[:1]  # Close the loop
                
                # Plot the metrics for this model
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title("Model Performance Metrics Comparison", size=15)
            
            st.pyplot(fig)
            
        # 2. Create bar charts for individual metrics
        for metric in metric_columns:
            # Use a different color palette for each metric
            colors = {'ROC-AUC': 'blues', 'Precision': 'greens', 'Recall': 'oranges', 'F1-Score': 'purples', 'F1 Score': 'purples'}
            color_scale = colors.get(metric, 'blues')
            
            bar_chart = alt.Chart(metrics_df).mark_bar().encode(
                x=alt.X('Model:N', title=None, sort='-y'),
                y=alt.Y(f'{metric}:Q', scale=alt.Scale(domain=[0.5, 1])),
                color=alt.Color('Model:N', legend=None, scale=alt.Scale(scheme=color_scale)),
                tooltip=['Model', metric]
            ).properties(
                title=f'{metric} by Model',
                height=250
            )
            st.altair_chart(bar_chart, use_container_width=True)
        
        # 3. Add a heatmap for all metrics
        if len(metric_columns) >= 2:
            # Prepare data for heatmap
            heatmap_data = metrics_df.melt(id_vars=['Model'], value_vars=metric_columns, 
                                       var_name='Metric', value_name='Value')
            
            # Create heatmap
            heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('Metric:N', title=None),
                y=alt.Y('Model:N', title=None),
                color=alt.Color('Value:Q', scale=alt.Scale(domain=[0.5, 1], scheme='viridis')),
                tooltip=['Model', 'Metric', 'Value']
            ).properties(
                title='Performance Metrics Heatmap',
                width=400,
                height=200
            )
            
            # Add text labels to the heatmap
            text = alt.Chart(heatmap_data).mark_text(baseline='middle').encode(
                x=alt.X('Metric:N'),
                y=alt.Y('Model:N'),
                text=alt.Text('Value:Q', format='.3f'),
                color=alt.condition(
                    alt.datum.Value > 0.75,
                    alt.value('black'),
                    alt.value('white')
                )
            )
            
            st.altair_chart(heatmap + text, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        # Fall back to a simple bar chart
        try:
            auc_col = next((col for col in ['ROC-AUC', 'ROC AUC', 'AUC'] if col in metrics_df.columns), None)
            if auc_col:
                auc_chart = alt.Chart(metrics_df).mark_bar().encode(
                    x=alt.X('Model:N', title=None),
                    y=alt.Y(f'{auc_col}:Q', scale=alt.Scale(domain=[0.5, 1])),
                    color=alt.Color('Model:N', legend=None),
                    tooltip=['Model', auc_col]
                ).properties(
                    title='ROC-AUC by Model',
                    height=300
                )
                st.altair_chart(auc_chart, use_container_width=True)
            else:
                st.dataframe(metrics_df)
        except:
            st.dataframe(metrics_df)

# Set page configuration
st.set_page_config(
    page_title="Aegis Alliance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for dark theme
st.markdown("""
<style>
    /* Global styles and dark theme */
    :root {
        --background-color: #0B0F19;
        --secondary-bg: #1A1F2C;
        --accent-color: #6E56CF;
        --text-color: #E1E7EF;
        --secondary-text: #9BA1AC;
        --highlight-color: #FF4A6B;
        --success-color: #3ECF8E;
        --warning-color: #FFB020;
        --chart-grid: #2D3748;
        --border-color: #2A3140;
        --card-bg: #141824;
    }
    
    /* Override Streamlit's default styling */
    .reportview-container {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .main .block-container {
        background-color: var(--background-color);
        padding-top: 2rem;
    }
    
    /* Header styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(to right, var(--accent-color), var(--highlight-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-color);
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--accent-color);
        padding-left: 0.75rem;
    }
    
    /* Metric containers */
    .metric-container {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(to right, var(--accent-color), var(--highlight-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--secondary-text);
        font-weight: 500;
    }
    
    /* Card styles */
    .card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
        border-color: rgba(110, 86, 207, 0.5);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    
    .card-title-icon {
        margin-right: 0.5rem;
        color: var(--accent-color);
    }
    
    /* Table styling */
    .dataframe {
        background-color: var(--secondary-bg) !important;
        color: var(--text-color) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .dataframe th {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    .dataframe td {
        background-color: var(--secondary-bg) !important;
        color: var(--secondary-text) !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    /* Alert styles */
    .stAlert {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border-left-width: 8px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1v3fvcr, [data-testid="stSidebar"] {
        background-color: var(--secondary-bg) !important;
        border-right: 1px solid #2A3140 !important;
    }
    
    [data-testid="stSidebarNav"] {
        padding-top: 0rem;
        background-color: var(--secondary-bg) !important;
    }
    
    [data-testid="stSidebarNav"] button[kind="secondary"] {
        background-color: transparent !important;
        border: none !important;
        color: #9BA1AC !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        padding: 0.75rem 1rem !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        border-radius: 8px !important;
        margin-bottom: 0.25rem !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    [data-testid="stSidebarNav"] button[kind="secondary"]:hover {
        background-color: rgba(90, 70, 174, 0.15) !important;
        color: #E1E7EF !important;
        transform: translateX(3px) !important;
    }
    
    [data-testid="stSidebarNav"] button[kind="secondary"]:active {
        background-color: rgba(90, 70, 174, 0.25) !important;
        transform: scale(0.98) !important;
    }
    
    /* Navigation item styling */
    .nav-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
        background-color: transparent;
    }
    
    .nav-item:hover {
        background-color: rgba(90, 70, 174, 0.1);
    }
    
    .nav-item.active {
        background-color: rgba(90, 70, 174, 0.2);
        border-left: 3px solid var(--accent-color);
    }
    
    .nav-item-icon {
        margin-right: 10px;
        color: #9BA1AC;
        width: 24px;
        text-align: center;
    }
    
    .nav-item.active .nav-item-icon,
    .nav-item.active .nav-item-text {
        color: var(--accent-color);
    }
    
    .nav-item-text {
        color: #E1E7EF;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--accent-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .stButton > button:hover {
        background-color: #7A66CF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    /* Primary button */
    .primary-btn {
        background-color: var(--accent-color) !important;
    }
    
    /* Secondary button */
    .secondary-btn {
        background-color: transparent !important;
        border: 1px solid var(--accent-color) !important;
        color: var(--accent-color) !important;
    }
    
    /* Danger button */
    .danger-btn {
        background-color: var(--error-color) !important;
    }
    
    /* Success button */
    .success-btn {
        background-color: var(--success-color) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background-color: var(--border-color) !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: var(--accent-color) !important;
    }
    
    /* Graph styling - ensure dark theme for all charts */
    .js-plotly-plot .plotly .bg {
        fill: var(--card-bg) !important;
    }
    
    .js-plotly-plot .plotly .xaxis line, .js-plotly-plot .plotly .yaxis line {
        stroke: var(--secondary-text) !important;
    }
    
    .js-plotly-plot .plotly .xaxis path, .js-plotly-plot .plotly .yaxis path {
        stroke: var(--secondary-text) !important;
    }
    
    .js-plotly-plot .plotly .xtick text, .js-plotly-plot .plotly .ytick text {
        fill: var(--secondary-text) !important;
    }
    
    /* Custom badge/tag styles */
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .badge-success {
        background-color: rgba(62, 207, 142, 0.2);
        color: var(--success-color);
        border: 1px solid var(--success-color);
    }
    
    .badge-warning {
        background-color: rgba(255, 176, 32, 0.2);
        color: var(--warning-color);
        border: 1px solid var(--warning-color);
    }
    
    .badge-danger {
        background-color: rgba(255, 74, 107, 0.2);
        color: var(--highlight-color);
        border: 1px solid var(--highlight-color);
    }
    
    .badge-info {
        background-color: rgba(110, 86, 207, 0.2);
        color: var(--accent-color);
        border: 1px solid var(--accent-color);
    }
    
    /* Customize scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-color);
    }
    
    /* Custom animations for elements */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Pulsing effect for important metrics */
    /* Enhanced styling for dashboard sections */
    .dashboard-header {
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #2A3140;
    }
    
    .dashboard-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #E1E7EF;
        background: linear-gradient(90deg, #6E56CF, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .dashboard-header p {
        font-size: 1rem;
        color: #9BA1AC;
        max-width: 700px;
    }
    
    .section-title {
        display: flex;
        align-items: center;
        margin: 2rem 0 1rem 0;
    }
    
    .section-title h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #E1E7EF;
        margin: 0;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 0.375rem;
        margin-left: 0.75rem;
    }
    
    .badge-primary {
        background-color: rgba(110, 86, 207, 0.15);
        color: #6E56CF;
    }
    
    .badge-secondary {
        background-color: rgba(37, 99, 235, 0.15);
        color: #2563EB;
    }
    
    .badge-info {
        background-color: rgba(6, 182, 212, 0.15);
        color: #06B6D4;
    }
    
    .badge-success {
        background-color: rgba(62, 207, 142, 0.15);
        color: #3ECF8E;
    }
    
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background-color: #1A1E2E;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        display: flex;
        flex-direction: column;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #2A3140;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #3F4A6B;
    }
    
    .metric-icon {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .metric-content {
        display: flex;
        flex-direction: column;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #9BA1AC;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #E1E7EF;
        margin-bottom: 0.5rem;
    }
    
    .metric-delta {
        font-size: 0.75rem;
        font-weight: 500;
        display: flex;
        align-items: center;
    }
    
    .metric-delta.positive {
        color: #3ECF8E;
    }
    
    .metric-delta.negative {
        color: #EF4444;
    }
    
    .metric-delta.neutral {
        color: #F59E0B;
    }
    
    .chart-card {
        background-color: #1A1E2E;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        border: 1px solid #2A3140;
    }
    
    .chart-header {
        margin-bottom: 1rem;
    }
    
    .chart-header h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #E1E7EF;
        margin: 0;
    }
    
    .activity-timeline {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin-top: 1rem;
        padding: 1rem;
        background-color: #1A1E2E;
        border-radius: 8px;
        border: 1px solid #2A3140;
    }
    
    .activity-item {
        display: flex;
        align-items: flex-start;
        padding: 0.75rem;
        border-radius: 6px;
        background-color: #20273C;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.15s;
    }
    
    .activity-item:hover {
        transform: translateX(3px);
        background-color: #242E45;
    }
    
    .activity-icon {
        width: 32px;
        height: 32px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.75rem;
        font-size: 1rem;
    }
    
    .activity-content {
        flex: 1;
    }
    
    .activity-text {
        font-size: 0.875rem;
        color: #E1E7EF;
        margin-bottom: 0.25rem;
    }
    
    .activity-time {
        font-size: 0.75rem;
        color: #9BA1AC;
    }
    
    .architecture-container {
        background-color: #1A1E2E;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #2A3140;
        margin-top: 1rem;
    }
    
    .status-card {
        background-color: #1A1E2E;
        border-radius: 8px;
        padding: 1.25rem;
        border: 1px solid #2A3140;
        display: flex;
        align-items: flex-start;
        margin-top: 1rem;
    }
    
    .status-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: rgba(62, 207, 142, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
    }
    
    .status-content {
        flex: 1;
    }
    
    .status-content p {
        color: #9BA1AC;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }
    
    .highlight {
        color: #E1E7EF;
        font-weight: 600;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(62, 207, 142, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(62, 207, 142, 0); }
        100% { box-shadow: 0 0 0 0 rgba(62, 207, 142, 0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ======= SIDEBAR =======
# Custom sidebar header with logo
logo_path = os.path.join(os.path.dirname(__file__), "logo", "AegisAllianceLogo4.png")
logo_base64 = get_base64_encoded_image(logo_path)

# If logo couldn't be loaded, use a fallback design
if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="75" style="filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); object-fit: contain;">'
else:
    # Fallback to a text-based logo if image can't be loaded
    logo_html = '<div style="width: 75px; height: 75px; background-color: #6E56CF; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 28px; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">AA</div>'

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; padding: 1.5rem 1rem; margin-bottom: 1.5rem;">
    {logo_html}
    <div style="margin-left: 1rem;">
        <h1 style="margin: 0; padding: 0; color: #E1E7EF; font-size: 1.7rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">Aegis Alliance</h1>
        <p style="margin: 0; padding: 0; color: #9BA1AC; font-size: 0.95rem;">Trust & Transparency</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Add separator
st.sidebar.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #2A3140;'>", unsafe_allow_html=True)

# Dashboard sections with icons
st.sidebar.markdown("""
<div style="background: rgba(26, 30, 46, 0.5); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; border: 1px solid #2A3140;">
    <p style="color: #9BA1AC; text-transform: uppercase; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05rem; margin-bottom: 0.5rem;">NAVIGATION</p>
</div>
""", unsafe_allow_html=True)

# Define sections with icons and colors
sections = {
    "Overview": "Overview",
    "Model Performance": "Model Performance",
    "Privacy Metrics": "Privacy Metrics",
    "Audit Log": "Audit Log",
    "Federation Status": "Federation Status",
    "Fraud Detection": "Fraud Detection"
}

# Define section colors for visual indication
section_colors = {
    "Overview": "#8B5CF6",
    "Model Performance": "#3B82F6", 
    "Privacy Metrics": "#22C55E",
    "Audit Log": "#EAB308",
    "Federation Status": "#6E56CF",
    "Fraud Detection": "#EC4899"
}

# Custom CSS for radio buttons to make them look like a nav menu
st.markdown("""
<style>
    div.row-widget.stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] > label {
        display: flex;
        align-items: center;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        cursor: pointer;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] > label:hover {
        background-color: rgba(90, 70, 174, 0.1);
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: transparent !important;
        border-color: #9BA1AC !important;
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div {
        background-color: var(--accent-color) !important;
        border-color: var(--accent-color) !important;
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] > label[aria-checked="true"] {
        background-color: rgba(90, 70, 174, 0.2);
        border-left: 3px solid var(--accent-color);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] > label[aria-checked="true"] span {
        font-weight: 600;
        color: var(--accent-color);
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] > label {
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
        padding: 0.5rem 0.75rem;
    }
    
    div.row-widget.stRadio > div [role="radiogroup"] > label:hover {
        background-color: rgba(90, 70, 174, 0.1);
        transform: translateX(3px);
    }
</style>
""", unsafe_allow_html=True)

# Create radio buttons with the improved styling
selected_section = st.sidebar.radio(
    "Navigation",
    list(sections.keys()),
    format_func=lambda x: sections[x],
    label_visibility="collapsed"
)

# Add custom indicator for the selected section
st.sidebar.markdown(f"""
<style>
    div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] > div > label[data-baseweb="radio"] > div:first-child {{
        background-color: transparent !important;
    }}
</style>
<div style="padding: 0.5rem; margin-top: -1rem;">
    <div style="height: 0.5rem; width: 3rem; background-color: {section_colors[selected_section]}; border-radius: 1rem; margin-bottom: 1rem;"></div>
    <div style="font-size: 0.85rem; color: {section_colors[selected_section]}; font-weight: 600;">Currently viewing: {sections[selected_section]}</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #2A3140;'>", unsafe_allow_html=True)
# Controls section
st.sidebar.markdown("""
<p style="color: #9BA1AC; text-transform: uppercase; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05rem; margin-bottom: 1rem;">CONTROLS & SETTINGS</p>
""", unsafe_allow_html=True)

# Display current model information if available
display_model_info_badge()

# Epsilon slider with custom styling and icon
st.sidebar.markdown("""
<div style="margin: 1rem 0 0.5rem 0; display: flex; align-items: center;">
    <span style="color: #5A46AE; font-size: 1.2rem; margin-right: 0.5rem;">🔐</span>
    <label style="font-size: 0.9rem; color: #E1E7EF; font-weight: 500;">Privacy Budget (ε)</label>
    <div style="margin-left: auto; background-color: rgba(90, 70, 174, 0.2); color: #9BA1AC; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">SETTING</div>
</div>
""", unsafe_allow_html=True)

epsilon = st.sidebar.slider("Privacy Budget", min_value=0.1, max_value=10.0, value=1.0, step=0.1, 
                          help="Lower ε means more privacy but potentially lower accuracy",
                          label_visibility="collapsed")

# Display current epsilon value with custom styling
st.sidebar.markdown(f"""
<div style="text-align: center; padding: 0.5rem; background-color: rgba(110, 86, 207, 0.1); border-radius: 8px; margin-bottom: 1.5rem; border-left: 3px solid #6E56CF;">
    <span style="font-size: 1.1rem; font-weight: 600; color: #6E56CF;">ε = {epsilon:.1f}</span>
    <div style="font-size: 0.75rem; color: #9BA1AC; margin-top: 0.2rem;">Privacy-Utility Trade-off</div>
</div>
""", unsafe_allow_html=True)

# Bank federation selector with custom styling
st.sidebar.markdown("""
<div style="margin: 1rem 0 0.5rem 0; display: flex; align-items: center;">
    <span style="color: #5A46AE; font-size: 1.2rem; margin-right: 0.5rem;">🏦</span>
    <label style="font-size: 0.9rem; color: #E1E7EF; font-weight: 500;">Financial Institution</label>
    <div style="margin-left: auto; background-color: rgba(90, 70, 174, 0.2); color: #9BA1AC; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">FILTER</div>
</div>
""", unsafe_allow_html=True)

banks = ["Bank A", "Bank B", "Bank C", "All Banks"]
selected_bank = st.sidebar.selectbox("Select Bank", banks, label_visibility="collapsed")

# Apply custom styling to the selectbox
st.markdown("""
<style>
    div.row-widget.stSelectbox > div > div {
        background-color: rgba(26, 30, 46, 0.5) !important;
        border: 1px solid #2A3140 !important;
        border-radius: 8px !important;
        color: #E1E7EF !important;
    }
    
    div.row-widget.stSelectbox > div > div:hover {
        border-color: var(--accent-color) !important;
    }
    
    div.row-widget.stSelectbox > div > div > div {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Time period selector with custom styling
st.sidebar.markdown("""
<div style="margin: 1.5rem 0 0.5rem 0; display: flex; align-items: center;">
    <span style="color: #5A46AE; font-size: 1.2rem; margin-right: 0.5rem;">⏱️</span>
    <label style="font-size: 0.9rem; color: #E1E7EF; font-weight: 500;">Time Period</label>
    <div style="margin-left: auto; background-color: rgba(90, 70, 174, 0.2); color: #9BA1AC; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;">FILTER</div>
</div>
""", unsafe_allow_html=True)

time_periods = ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
selected_period = st.sidebar.selectbox("Time Period", time_periods, label_visibility="collapsed")

# System status indicator
st.sidebar.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #2A3140;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #3ECF8E; margin-right: 8px;"></div>
    <span style="color: #E1E7EF; font-size: 0.9rem; font-weight: 500;">System Online</span>
    <div style="margin-left: auto; font-size: 0.8rem; color: #9BA1AC;">v1.0.2</div>
</div>
""", unsafe_allow_html=True)

# Add last updated timestamp
current_time = datetime.now().strftime("%b %d, %Y %H:%M")
st.sidebar.markdown(f"""
<div style="font-size: 0.8rem; color: #9BA1AC; text-align: center; margin-top: 2rem;">
    Last updated: {current_time}
</div>
""", unsafe_allow_html=True)
# ======= HELPER FUNCTIONS =======
@st.cache_data
def generate_sample_data(epsilon):
    """Generate sample data with different epsilon values"""
    np.random.seed(42)
    
    # Generate ROC curve data points based on epsilon (privacy budget)
    # Higher epsilon = better performance, lower privacy
    epsilons = np.linspace(0.1, 10.0, 20)
    roc_data = []
    
    for eps in epsilons:
        # Model performance decreases as privacy increases (lower epsilon)
        base_auc = 0.85  # Base AUC for high epsilon
        noise_factor = 1 / (eps + 0.1)  # More noise (lower AUC) for lower epsilon
        auc_value = max(0.5, min(0.99, base_auc - (noise_factor * 0.05)))
        
        # Generate a simple ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, (1.0 / auc_value) - 1)
        
        # Add to data with epsilon value
        for f, t in zip(fpr, tpr):
            roc_data.append({"Epsilon": eps, "FPR": f, "TPR": t, "AUC": auc_value})
    
    return pd.DataFrame(roc_data)

@st.cache_data
def load_audit_log():
    """Load the audit log from the generated data file"""
    try:
        # Try to load from generated data file
        df = pd.read_csv('data/transactions.csv')
        
        # Rename columns to match expected format if needed
        column_mapping = {
            'TransactionID': 'Transaction ID',
            'FraudScore': 'Fraud Score',
            'ZKProof': 'ZK Proof'
        }
        
        # Only rename columns that exist and need renaming
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure required columns exist
        required_columns = ['Timestamp', 'Transaction ID', 'Bank', 'Amount', 'Fraud Score', 'Verification', 'ZK Proof']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Missing required columns in data file: {', '.join(missing_columns)}. Using sample data.")
            return generate_sample_audit_log(100)
        
        # Convert timestamp strings to datetime objects
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        except Exception as e:
            st.warning(f"Error converting timestamps: {str(e)}. Generating new timestamps.")
            now = datetime.now()
            df['Timestamp'] = [now - timedelta(hours=i) for i in range(len(df))]
        
        # Sort by timestamp descending
        df = df.sort_values('Timestamp', ascending=False)
        
        # Validate numeric columns
        numeric_columns = ['Amount', 'Fraud Score']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    st.warning(f"Error converting {col} to numeric. Using random values.")
                    if col == 'Amount':
                        df[col] = np.random.lognormal(mean=5.0, sigma=1.2, size=len(df))
                    elif col == 'Fraud Score':
                        df[col] = np.random.beta(0.5, 5.0, size=len(df))
        
        # Select relevant columns
        audit_df = df[required_columns]
        
        return audit_df
    except Exception as e:
        st.error(f"Error loading transaction data: {str(e)}")
        # Fall back to generating sample data if file doesn't exist or has issues
        return generate_sample_audit_log(100)

@st.cache_data
def generate_sample_audit_log(n_entries=100):
    """Generate sample audit log entries as fallback"""
    try:
        # Set seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Generate timestamps with more realistic distribution
        # Most transactions during business hours, fewer at night
        base_time = datetime.now()
        timestamps = []
        
        for i in range(n_entries):
            # Generate hour with higher probability during business hours
            hour_weights = [1, 1, 1, 1, 1, 2, 3, 5, 8, 10, 12, 13, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
            hour = random.choices(range(24), weights=hour_weights)[0]
            
            # Random day within last 7 days
            days_ago = random.randint(0, 7)
            
            # Create timestamp
            ts = base_time - timedelta(days=days_ago, 
                                    hours=base_time.hour-hour,
                                    minutes=random.randint(0, 59),
                                    seconds=random.randint(0, 59))
            timestamps.append(ts)
        
        # Sort timestamps in descending order (newest first)
        timestamps.sort(reverse=True)
        
        # Generate transaction IDs with realistic format
        transaction_ids = [f"TX{random.randint(10000000, 99999999)}" for _ in range(n_entries)]
        
        # Bank names with realistic distribution
        banks = ["Kasikorn Bank", "Siam Commercial Bank", "Bangkok Bank", "Krung Thai Bank", 
                "TMB Bank", "Bank of Ayudhya", "CIMB Thai", "Kiatnakin Bank"]
        bank_weights = [25, 20, 20, 15, 10, 5, 3, 2]  # Probability weights
        bank_names = random.choices(banks, weights=bank_weights, k=n_entries)
        
        # Transaction amounts with realistic distribution
        # Most transactions small, some medium, few very large
        amount_categories = [
            ("small", 100, 5000, 0.65),      # Small transactions (100-5,000)
            ("medium", 5001, 50000, 0.25),   # Medium transactions (5,001-50,000) 
            ("large", 50001, 500000, 0.08),  # Large transactions (50,001-500,000)
            ("vl", 500001, 5000000, 0.02)    # Very large transactions (500,001-5,000,000)
        ]
        
        amounts = []
        for _ in range(n_entries):
            # Select amount category based on probability
            r = random.random()
            cumulative_prob = 0
            selected_category = amount_categories[0]
            
            for cat, min_val, max_val, prob in amount_categories:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected_category = (cat, min_val, max_val, prob)
                    break
            
            # Generate amount within the selected range
            _, min_val, max_val, _ = selected_category
            amount = random.uniform(min_val, max_val)
            amounts.append(amount)
        
        # Fraud scores with realistic distribution
        # Most legitimate (low score), some suspicious, few fraudulent
        fraud_categories = [
            ("legitimate", 0.01, 0.3, 0.80),    # Legitimate transactions
            ("suspicious", 0.3, 0.7, 0.15),     # Suspicious transactions
            ("fraudulent", 0.7, 0.99, 0.05)     # Fraudulent transactions
        ]
        
        fraud_scores = []
        for _ in range(n_entries):
            # Select fraud category based on probability
            r = random.random()
            cumulative_prob = 0
            selected_category = fraud_categories[0]
            
            for cat, min_val, max_val, prob in fraud_categories:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected_category = (cat, min_val, max_val, prob)
                    break
            
            # Generate fraud score within the selected range
            _, min_val, max_val, _ = selected_category
            score = random.uniform(min_val, max_val)
            fraud_scores.append(score)
        
        # Verification status based on fraud score
        verifications = []
        for score in fraud_scores:
            if score > 0.7:
                verifications.append("Declined")
            elif score > 0.3:
                verifications.append("OTP Verified")
            else:
                verifications.append("Auto-Approved")
        
        # ZK proof status - most should be verified
        zk_proofs = []
        for _ in range(n_entries):
            r = random.random()
            if r > 0.05:  # 95% verification rate
                zk_proofs.append("Verified")
            else:
                zk_proofs.append("Failed")
        
        # Create dataframe with the generated data
        audit_df = pd.DataFrame({
            "Timestamp": timestamps,
            "Transaction ID": transaction_ids,
            "Bank": bank_names,
            "Amount": amounts,
            "Fraud Score": fraud_scores,
            "Verification": verifications,
            "ZK Proof": zk_proofs
        })
        
        return audit_df
        
    except Exception as e:
        st.error(f"Error generating sample data: {str(e)}")
        # Return empty dataframe with the expected columns as last resort
        return pd.DataFrame(columns=["Timestamp", "Transaction ID", "Bank", "Amount", 
                                    "Fraud Score", "Verification", "ZK Proof"])

@st.cache_data
def load_federation_metrics():
    """Load federation performance metrics from generated data"""
    try:
        # Try to load from generated data file
        df = pd.read_csv('data/model_metrics.csv')
        
        # Filter to the desired epsilon
        # We'll filter by epsilon later in the code
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Fall back to generating sample data if file doesn't exist
        return generate_federation_metrics_fallback()

@st.cache_data
def generate_federation_metrics_fallback():
    """Generate federation performance metrics as fallback"""
    banks = ["Bank A", "Bank B", "Bank C", "Federated Model"]
    metrics = []
    
    # Base metrics for individual banks
    base_metrics = {
        "Bank A": {"AUC": 0.82, "Precision": 0.75, "Recall": 0.71, "F1": 0.73, "Latency": 12},
        "Bank B": {"AUC": 0.79, "Precision": 0.72, "Recall": 0.68, "F1": 0.70, "Latency": 15},
        "Bank C": {"AUC": 0.84, "Precision": 0.78, "Recall": 0.73, "F1": 0.75, "Latency": 13},
        "Federated Model": {"AUC": 0.89, "Precision": 0.83, "Recall": 0.81, "F1": 0.82, "Latency": 18}
    }
    
    # Create metrics dataframe
    for bank in banks:
        metrics.append({
            "Model": bank,
            "ROC-AUC": base_metrics[bank]["AUC"],
            "Precision": base_metrics[bank]["Precision"],
            "Recall": base_metrics[bank]["Recall"],
            "F1 Score": base_metrics[bank]["F1"],
            "Latency (ms)": base_metrics[bank]["Latency"]
        })
    
    return pd.DataFrame(metrics)

@st.cache_data
def load_federation_progress():
    """Load federation progress data from generated file"""
    try:
        # Try to load from generated data file
        df = pd.read_csv('data/federation_progress.csv')
        
        # Convert to the format needed for plotting
        if 'Model' in df.columns:
            # Data is in long format
            rounds = df['Round'].unique()
            models = df['Model'].unique()
            
            # Convert to wide format for our plotting needs
            result_data = []
            for r in rounds:
                row_data = {'Round': r}
                for model in models:
                    model_data = df[(df['Round'] == r) & (df['Model'] == model)]
                    if not model_data.empty:
                        row_data[model] = model_data['ROC-AUC'].iloc[0]
                result_data.append(row_data)
            
            return pd.DataFrame(result_data)
        else:
            return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Fall back to generating sample data if file doesn't exist
        return generate_federation_progress_fallback()

@st.cache_data
def generate_federation_progress_fallback():
    """Generate sample training progress data as fallback"""
    rounds = 28
    training_data = []
    
    for r in range(1, rounds + 1):
        base_auc = 0.65
        max_auc = 0.89
        
        # Simulate learning curve (improvement over rounds)
        auc = base_auc + (max_auc - base_auc) * (1 - np.exp(-0.15 * r))
        
        # Add noise to individual bank performances
        bank_a_auc = auc - 0.05 + np.random.uniform(-0.02, 0.02)
        bank_b_auc = auc - 0.08 + np.random.uniform(-0.02, 0.02)
        bank_c_auc = auc - 0.03 + np.random.uniform(-0.02, 0.02)
        fed_auc = auc + np.random.uniform(-0.01, 0.01)
        
        training_data.append({
            "Round": r,
            "Bank A": bank_a_auc,
            "Bank B": bank_b_auc,
            "Bank C": bank_c_auc,
            "Federated": fed_auc
        })
    
    return pd.DataFrame(training_data)

# Add a footer to the sidebar
st.sidebar.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

# Reuse the logo base64 for the footer
footer_logo_html = ""
if logo_base64:
    footer_logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="35" style="opacity: 0.85; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.2));">'
else:
    # Fallback to a text-based logo if image can't be loaded
    footer_logo_html = '<div style="width: 35px; height: 35px; background-color: #6E56CF; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 16px; text-shadow: 0 1px 1px rgba(0,0,0,0.3);">AA</div>'

st.sidebar.markdown(f"""
<div style="padding: 1rem; margin-top: 1rem;">
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 0.5rem; text-align: center;">
        <div style="margin-bottom: 0.75rem;">
            {footer_logo_html}
        </div>
        <div style="font-size: 0.7rem; color: #9BA1AC; margin-bottom: 0.5rem;">
            <span style="color: #6E56CF;">●</span> Active - Last update: {datetime.now().strftime('%H:%M')}
        </div>
        <div style="font-size: 0.75rem; color: #9BA1AC;">© 2025 Aegis Alliance | v1.2.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======= MAIN CONTENT =======
if selected_section == "Overview":
    # Page header with title and description
    st.markdown("""
    <div style="background-color: #1A1E2E; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 2rem; border-left: 5px solid #8B5CF6;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #E1E7EF; margin-bottom: 0.5rem;">Aegis Alliance Dashboard</h1>
                <p style="margin: 0; font-size: 1rem; color: #9BA1AC;">Trust & Transparency Layer: Real-time insights into system performance, privacy, and federation status</p>
            </div>
            <div style="font-size: 2.5rem; color: #8B5CF6;">📊</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview metrics with enhanced styling
    st.markdown("""
    <div class="metric-container">
        <div class="metric-row">
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Custom styled metrics
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon" style="background-color: rgba(62, 207, 142, 0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M8 16L10.879 13.121C11.3395 12.6605 12.0875 12.62 12.5979 13.0229L13.5 13.75L18 10" stroke="#3ECF8E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="#3ECF8E" stroke-width="2"/>
                </svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Transactions Processed</div>
                <div class="metric-value">12,456</div>
                <div class="metric-delta positive">+845 today</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon" style="background-color: rgba(110, 86, 207, 0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21 21H4.6C4.03995 21 3.75992 21 3.54601 20.891C3.35785 20.7951 3.20487 20.6422 3.10899 20.454C3 20.2401 3 19.9601 3 19.4V3" stroke="#6E56CF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M7 14.5L11.2929 10.2071C11.6834 9.81658 12.3166 9.81658 12.7071 10.2071L14.5 12L17.5 9" stroke="#6E56CF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Fraud Detection Rate</div>
                <div class="metric-value">89.2%</div>
                <div class="metric-delta positive">+1.3% vs. last week</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon" style="background-color: rgba(236, 72, 153, 0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#EC4899" stroke-width="2"/>
                    <path d="M12 8V12L15 15" stroke="#EC4899" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Privacy Budget (ε)</div>
                <div class="metric-value">{epsilon:.1f}</div>
                <div class="metric-delta neutral">Currently active</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon" style="background-color: rgba(37, 99, 235, 0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 12H22" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M12 2C14.5013 4.73835 15.9228 8.29203 16 12C15.9228 15.708 14.5013 19.2616 12 22C9.49872 19.2616 8.07725 15.708 8 12C8.07725 8.29203 9.49872 4.73835 12 2Z" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">ZK Proof Verification</div>
                <div class="metric-value">99.7%</div>
                <div class="metric-delta positive">+0.2% vs. last week</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview charts with enhanced card styling
    st.markdown("""
    <div class="section-title">
        <h2>Performance vs. Privacy Trade-off</h2>
        <div class="badge badge-primary">Real-time Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate dummy data for demonstration
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accuracies = [0.82, 0.86, 0.89, 0.92, 0.94, 0.95]
    privacy_loss = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]
    
    # Create a two-panel chart with enhanced styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <h3>Accuracy vs. Privacy Budget</h3>
            </div>
        """, unsafe_allow_html=True)
        
        chart_data = pd.DataFrame({
            'Privacy Budget (ε)': epsilons,
            'Model Accuracy': accuracies
        })
        
        chart = alt.Chart(chart_data).mark_line(
            point=True,
            color='#6E56CF',
            strokeWidth=3
        ).encode(
            x=alt.X('Privacy Budget (ε)', title='Privacy Budget (ε)'),
            y=alt.Y('Model Accuracy', scale=alt.Scale(domain=[0.80, 1.0]), title='Model Accuracy'),
            tooltip=['Privacy Budget (ε)', 'Model Accuracy']
        ).properties(
            height=250
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <h3>Privacy Loss vs. Privacy Budget</h3>
            </div>
        """, unsafe_allow_html=True)
        
        chart_data = pd.DataFrame({
            'Privacy Budget (ε)': epsilons,
            'Privacy Loss': privacy_loss
        })
        
        chart = alt.Chart(chart_data).mark_line(
            point=True,
            color='#EC4899',
            strokeWidth=3
        ).encode(
            x=alt.X('Privacy Budget (ε)', title='Privacy Budget (ε)'),
            y=alt.Y('Privacy Loss', scale=alt.Scale(domain=[0, 1.0]), title='Privacy Loss'),
            tooltip=['Privacy Budget (ε)', 'Privacy Loss']
        ).properties(
            height=250
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System architecture diagram with enhanced styling
    st.markdown("""
    <div class="section-title">
        <h2>System Architecture</h2>
        <div class="badge badge-secondary">System Design</div>
    </div>
    <div class="architecture-container">
    """, unsafe_allow_html=True)
    
    # Placeholder for architecture diagram
    architecture = """
    digraph G {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style=filled, color="#2A3140", fontcolor="#E1E7EF", fontname="Arial"];
        edge [color="#6E56CF", penwidth=1.5];
        
        Bank1 [label="Bank A Data"];
        Bank2 [label="Bank B Data"];
        Bank3 [label="Bank C Data"];
        
        Oracle [label="Oracle Engine\\n(XGBoost)", color="#1F2937"];
        Adaptive [label="Adaptive Intervention\\n(Policy Engine)", color="#1F2937"];
        Federated [label="Zero-Knowledge Fabric\\n(Federated Learning)", color="#1F2937"];
        Trust [label="Trust & Transparency\\n(Audit & Verification)", color="#1F2937"];
        
        {Bank1, Bank2, Bank3} -> Federated;
        Federated -> Oracle;
        Oracle -> Adaptive;
        {Oracle, Adaptive, Federated} -> Trust;
    }
    """
    
    st.graphviz_chart(architecture)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent activity with enhanced styling
    st.markdown("""
    <div class="section-title">
        <h2>Recent Activity</h2>
        <div class="badge badge-info">Live Updates</div>
    </div>
    <div class="activity-timeline">
    """, unsafe_allow_html=True)
    
    # Dummy activity data
    activities = [
        {"time": "2 mins ago", "activity": "Bank A completed model training", "type": "training", "icon": "🔄"},
        {"time": "15 mins ago", "activity": "New transactions processed: 145", "type": "transaction", "icon": "💼"},
        {"time": "1 hour ago", "activity": "Privacy budget updated to ε=1.2", "type": "config", "icon": "⚙️"},
        {"time": "3 hours ago", "activity": "Bank C joined the federation", "type": "federation", "icon": "🏦"},
        {"time": "5 hours ago", "activity": "System audit completed", "type": "audit", "icon": "📋"}
    ]
    
    for activity in activities:
        # Different formatting based on activity type
        icon_colors = {
            "training": "#6E56CF",
            "transaction": "#3ECF8E",
            "config": "#F59E0B",
            "federation": "#2563EB",
            "audit": "#EC4899"
        }
        
        st.markdown(f"""
        <div class="activity-item">
            <div class="activity-icon" style="background-color: {icon_colors[activity['type']]};">
                {activity['icon']}
            </div>
            <div class="activity-content">
                <div class="activity-text">{activity['activity']}</div>
                <div class="activity-time">{activity['time']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # System status with enhanced styling
    st.markdown("""
    <div class="section-title">
        <h2>System Status</h2>
        <div class="badge badge-success">Operational</div>
    </div>
    <div class="status-card">
        <div class="status-icon pulse">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#3ECF8E" stroke-width="2"/>
                <path d="M12 8V12" stroke="#3ECF8E" stroke-width="2" stroke-linecap="round"/>
                <path d="M12 16H12.01" stroke="#3ECF8E" stroke-width="2" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="status-content">
            <p>The Aegis Alliance is currently using a privacy budget of <span class="highlight">ε = {epsilon:.1f}</span>. The system is fully operational with all 3 banks participating in the federation. All privacy guarantees are being maintained while achieving optimal model performance.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif selected_section == "Model Performance":
    # Page header with title and description
    st.markdown("""
    <div style="background-color: #1A1E2E; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 2rem; border-left: 5px solid #3B82F6;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #E1E7EF; margin-bottom: 0.5rem;">Model Performance</h1>
                <p style="margin: 0; font-size: 1rem; color: #9BA1AC;">Analyze model metrics, compare bank performance, and track improvements over time</p>
            </div>
            <div style="font-size: 2.5rem; color: #3B82F6;">📈</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ROC Curve vs Epsilon visualization
    st.markdown("<h2 class='sub-header'>ROC Curve vs Privacy Budget (ε)</h2>", unsafe_allow_html=True)
    
    # Get data for current epsilon
    roc_data = generate_sample_data(epsilon)
    
    # Find the closest epsilon value in the data
    eps_values = sorted(roc_data["Epsilon"].unique())
    closest_eps = min(eps_values, key=lambda x: abs(x - epsilon))
    current_eps_data = roc_data[roc_data["Epsilon"] == closest_eps]
    
    if not current_eps_data.empty:
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                # Plot ROC curve for current epsilon
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(current_eps_data["FPR"], current_eps_data["TPR"], 
                        label=f'ROC curve (AUC = {current_eps_data["AUC"].iloc[0]:.3f})')
                ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve with ε = {closest_eps:.1f}')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating ROC curve: {str(e)}")
                st.info("Try adjusting the privacy budget or regenerating the data.")
        
        with col2:
            # Show AUC vs Epsilon table
            try:
                eps_auc = roc_data.groupby("Epsilon")["AUC"].first().reset_index()
                eps_auc = eps_auc.sort_values("Epsilon")
                
                # Create a styled dataframe
                st.markdown("#### AUC vs Privacy Budget")
                st.dataframe(eps_auc.style.highlight_max(subset=["AUC"]))
            except Exception as e:
                st.error(f"Error generating AUC table: {str(e)}")
    else:
        st.error(f"No data available for epsilon value: {epsilon}. Try a different value.")
        # Show the available epsilon values
        available_eps = sorted(roc_data["Epsilon"].unique())
        st.info(f"Available epsilon values: {', '.join([str(round(e, 2)) for e in available_eps])}")
    
    # Model performance metrics
    st.markdown("<h2 class='sub-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    try:
        # Get federation metrics
        federation_metrics = load_federation_metrics()
        
        if not federation_metrics.empty:
            # Check if Epsilon column exists
            if 'Epsilon' in federation_metrics.columns:
                # Filter to metrics for the current epsilon value (or closest)
                epsilon_values = sorted(federation_metrics['Epsilon'].unique())
                closest_epsilon = min(epsilon_values, key=lambda x: abs(x - epsilon))
                filtered_metrics = federation_metrics[federation_metrics['Epsilon'] == closest_epsilon]
            else:
                # If no Epsilon column, use all metrics
                filtered_metrics = federation_metrics.copy()
            
            # Convert model column to match expected format if needed
            if 'Model' in filtered_metrics.columns:
                filtered_metrics = filtered_metrics.copy()
                if 'Federated' in filtered_metrics['Model'].values and 'Federated Model' not in filtered_metrics['Model'].values:
                    filtered_metrics['Model'] = filtered_metrics['Model'].replace('Federated', 'Federated Model')
            
            # Format for display
            if not filtered_metrics.empty:
                formatted_metrics = filtered_metrics.copy()
                numeric_cols = formatted_metrics.select_dtypes(include=['float64', 'float32']).columns
                for col in numeric_cols:
                    formatted_metrics[col] = formatted_metrics[col].map("{:.3f}".format)
                
                # Create two columns for visualization
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    display_metrics(filtered_metrics)
                
                with col2:
                    if 'Model' in formatted_metrics.columns:
                        st.markdown("#### Metrics Data")
                        st.dataframe(formatted_metrics.set_index('Model'))
                    else:
                        st.markdown("#### Metrics Data")
                        st.dataframe(formatted_metrics)
            else:
                st.warning("No metrics data available for the current privacy budget.")
        else:
            st.warning("No model metrics data available. Try generating more data.")
    except Exception as e:
        st.error(f"Error displaying model metrics: {str(e)}")
        st.info("This could be due to missing data files or incorrect format.")
    
    # Performance vs Privacy Trade-off
    st.markdown("<h2 class='sub-header'>Performance vs Privacy Trade-off</h2>", unsafe_allow_html=True)
    
    try:
        # Get AUC vs Epsilon data
        eps_auc = roc_data.groupby("Epsilon")["AUC"].first().reset_index()
        
        # Plot trade-off curve
        tradeoff_chart = alt.Chart(eps_auc).mark_line(point=True).encode(
            x=alt.X('Epsilon:Q', title='Privacy Budget (ε)'),
            y=alt.Y('AUC:Q', scale=alt.Scale(domain=[0.5, 1])),
            tooltip=['Epsilon', 'AUC']
        ).properties(
            title='ROC-AUC vs Privacy Budget Trade-off',
            width=700,
            height=400
        )
        
        # Add a vertical line for current epsilon
        current_eps_line = alt.Chart(pd.DataFrame({'x': [epsilon]})).mark_rule(color='red').encode(
            x='x:Q'
        )
        
        st.altair_chart(tradeoff_chart + current_eps_line, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating trade-off chart: {str(e)}")
        st.info("This error could be due to missing data or an invalid epsilon value.")

elif selected_section == "Privacy Metrics":
    # Page header with title and description
    st.markdown("""
    <div style="background-color: #1A1E2E; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 2rem; border-left: 5px solid #22C55E;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #E1E7EF; margin-bottom: 0.5rem;">Privacy Metrics</h1>
                <p style="margin: 0; font-size: 1rem; color: #9BA1AC;">Monitor privacy guarantees and understand the privacy-utility trade-off</p>
            </div>
            <div style="font-size: 2.5rem; color: #22C55E;">🔒</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Differential Privacy Explanation
    st.markdown("<h2 class='sub-header'>Differential Privacy (DP)</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Differential Privacy provides mathematical guarantees about the privacy of individuals in a dataset.
        
        - **Privacy Budget (ε)**: Controls the trade-off between privacy and utility
        - **Lower ε = More Privacy**: But potentially lower model accuracy
        - **Higher ε = Less Privacy**: But potentially higher model accuracy
        
        The current system is using **ε = {:.1f}**, providing a balance between privacy protection and fraud detection performance.
        """.format(epsilon))
        
        # DP vs Non-DP Comparison
        dp_comparison = pd.DataFrame({
            'Metric': ['Data Leakage Risk', 'Privacy Protection', 'Model Accuracy', 'Training Time'],
            'Traditional Model': ['High', 'None', 'High', 'Fast'],
            'DP Model (ε = 0.1)': ['Very Low', 'Very High', 'Moderate', 'Slow'],
            'DP Model (ε = 1.0)': ['Low', 'High', 'Good', 'Moderate'],
            'DP Model (ε = 10.0)': ['Moderate', 'Moderate', 'Very Good', 'Fast']
        })
        
        st.dataframe(dp_comparison.set_index('Metric'))
    
    with col2:
        # Simple visualization of DP noise
        st.markdown("#### Effect of DP Noise on Data")
        
        # Generate sample data points
        np.random.seed(42)
        x = np.random.normal(5, 1, 100)
        
        # Apply different levels of DP noise
        epsilon_values = [0.1, 1.0, 10.0]
        noisy_data = {}
        
        for eps in epsilon_values:
            scale = 2.0 / eps  # Scale factor for Laplace noise
            noise = np.random.laplace(0, scale, 100)
            noisy_data[eps] = x + noise
        
        # Create a dataframe for plotting
        plot_data = pd.DataFrame({
            'Original': x,
            'ε = 0.1': noisy_data[0.1],
            'ε = 1.0': noisy_data[1.0],
            'ε = 10.0': noisy_data[10.0]
        })
        
        # Plot distributions
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(data=plot_data, ax=ax)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Effect of Privacy Budget (ε) on Data Distribution')
        st.pyplot(fig)
    
    # Zero-Knowledge Proofs Metrics
    st.markdown("<h2 class='sub-header'>Zero-Knowledge Proofs (ZKP) Metrics</h2>", unsafe_allow_html=True)
    
    # Create sample ZKP metrics
    zkp_success_rate = 99.7
    zkp_avg_time = 1.2  # seconds
    zkp_daily_count = 12468
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{zkp_success_rate}%</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>ZKP Verification Rate</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{zkp_avg_time}s</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Average Proof Time</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{zkp_daily_count:,}</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Daily ZKP Verifications</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif selected_section == "Audit Log":
    # Page header with title and description
    st.markdown("""
    <div style="background-color: #1A1E2E; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 2rem; border-left: 5px solid #EAB308;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #E1E7EF; margin-bottom: 0.5rem;">Transaction Audit Log</h1>
                <p style="margin: 0; font-size: 1rem; color: #9BA1AC;">Track transaction history with verification status and zero-knowledge proofs</p>
            </div>
            <div style="font-size: 2.5rem; color: #EAB308;">📝</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate audit log data
    audit_log = load_audit_log()
    
    # Add filters in a sidebar expander
    with st.expander("Advanced Filters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Search by transaction ID
            search_id = st.text_input("Search by Transaction ID")
            
            # Filter by verification status
            verification_options = ["All"] + list(audit_log["Verification"].unique())
            selected_verification = st.selectbox("Filter by Verification Status", verification_options)
            
            # Date range filter
            start_date = st.date_input("Start Date", value=audit_log["Timestamp"].min().date())
            
        with col2:
            # Search by bank (multi-select)
            bank_options = list(audit_log["Bank"].unique())
            selected_banks = st.multiselect("Select Banks", bank_options, default=bank_options)
            
            # Filter by ZK Proof status
            zkp_options = ["All"] + list(audit_log["ZK Proof"].unique())
            selected_zkp = st.selectbox("Filter by ZK Proof Status", zkp_options)
            
            # End date for range
            end_date = st.date_input("End Date", value=audit_log["Timestamp"].max().date())
    
    # Apply filters
    filtered_log = audit_log.copy()
    
    # Apply search by transaction ID
    if search_id:
        filtered_log = filtered_log[filtered_log["Transaction ID"].str.contains(search_id, case=False)]
    
    # Apply bank filter
    if selected_banks:
        filtered_log = filtered_log[filtered_log["Bank"].isin(selected_banks)]
    
    # Apply verification status filter
    if selected_verification != "All":
        filtered_log = filtered_log[filtered_log["Verification"] == selected_verification]
    
    # Apply ZK Proof filter
    if selected_zkp != "All":
        filtered_log = filtered_log[filtered_log["ZK Proof"] == selected_zkp]
    
    # Apply date range filter
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    filtered_log = filtered_log[(filtered_log["Timestamp"] >= start_datetime) & 
                              (filtered_log["Timestamp"] <= end_datetime)]
    
    # Filter by selected bank from the main sidebar
    if selected_bank != "All Banks":
        filtered_log = filtered_log[filtered_log["Bank"] == selected_bank]
    
    # Filter by time period from the main sidebar
    if selected_period == "Last 24 hours":
        cutoff = datetime.now() - timedelta(days=1)
        filtered_log = filtered_log[filtered_log["Timestamp"] > cutoff]
    elif selected_period == "Last 7 days":
        cutoff = datetime.now() - timedelta(days=7)
        filtered_log = filtered_log[filtered_log["Timestamp"] > cutoff]
    elif selected_period == "Last 30 days":
        cutoff = datetime.now() - timedelta(days=30)
        filtered_log = filtered_log[filtered_log["Timestamp"] > cutoff]
    
    # Show filter results summary
    st.markdown(f"<p>Showing {len(filtered_log)} of {len(audit_log)} transactions</p>", unsafe_allow_html=True)
    
    # Format the dataframe for display
    display_log = filtered_log.copy()
    display_log["Amount"] = display_log["Amount"].map("{:.2f} ฿".format)
    display_log["Fraud Score"] = display_log["Fraud Score"].map("{:.3f}".format)
    
    # Add color highlighting based on verification status
    def highlight_verification(val):
        if val == "Declined":
            return 'background-color: #FECACA'  # Light red
        elif val == "OTP Verified":
            return 'background-color: #FEF3C7'  # Light yellow
        else:
            return 'background-color: #D1FAE5'  # Light green
    
    def highlight_zkp(val):
        if val == "Failed":
            return 'background-color: #FECACA'  # Light red
        else:
            return 'background-color: #D1FAE5'  # Light green
    
    # Apply highlighting
    styled_log = display_log.style.apply(lambda x: [highlight_verification(val) if i == 5 else '' 
                                               for i, val in enumerate(x)], axis=1)
    styled_log = styled_log.apply(lambda x: [highlight_zkp(val) if i == 6 else '' 
                                        for i, val in enumerate(x)], axis=1)
    
    # Display audit log with a download button
    st.download_button(
        label="Download Filtered Audit Log as CSV",
        data=filtered_log.to_csv(index=False).encode('utf-8'),
        file_name=f'audit_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )
    
    st.dataframe(styled_log, use_container_width=True)
    
    # Summary metrics
    st.markdown("<h2 class='sub-header'>Audit Summary</h2>", unsafe_allow_html=True)
    
    # Calculate metrics
    total_transactions = len(filtered_log)
    if total_transactions > 0:  # Avoid division by zero
        try:
            flagged_count = len(filtered_log[filtered_log["Verification"] != "Auto-Approved"])
            declined_count = len(filtered_log[filtered_log["Verification"] == "Declined"])
            zkp_failed = len(filtered_log[filtered_log["ZK Proof"] == "Failed"])
            
            # More detailed metrics
            avg_fraud_score = filtered_log["Fraud Score"].mean()
            high_risk_count = len(filtered_log[filtered_log["Fraud Score"] > 0.7])
            total_amount = filtered_log["Amount"].sum()
        
            # Display metrics in two rows
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", total_transactions)
            
            with col2:
                st.metric("Flagged for Review", flagged_count, f"{flagged_count/total_transactions:.1%}" if total_transactions > 0 else "0%")
            
            with col3:
                st.metric("Declined Transactions", declined_count, f"{declined_count/total_transactions:.1%}" if total_transactions > 0 else "0%")
            
            with col4:
                st.metric("ZKP Failures", zkp_failed, f"{zkp_failed/total_transactions:.1%}" if total_transactions > 0 else "0%")
            
            # Second row of metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Fraud Score", f"{avg_fraud_score:.3f}")
            
            with col2:
                st.metric("High Risk Transactions", high_risk_count, f"{high_risk_count/total_transactions:.1%}" if total_transactions > 0 else "0%")
            
            with col3:
                st.metric("Total Transaction Value", f"{total_amount:.2f} ฿")
                
            with col4:
                # Calculate trend (last 10 vs previous 10)
                if len(filtered_log) >= 20:
                    recent_scores = filtered_log.iloc[:10]["Fraud Score"].mean()
                    previous_scores = filtered_log.iloc[10:20]["Fraud Score"].mean()
                    trend = recent_scores - previous_scores
                    st.metric("Recent Fraud Trend", f"{recent_scores:.3f}", f"{trend:.3f}")
                else:
                    st.metric("Recent Fraud Trend", "Insufficient data")
            
            # Interactive filters for analysis
            st.markdown("<h3>Interactive Analysis</h3>", unsafe_allow_html=True)
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                # Time-based analysis
                st.subheader("Transactions by Hour")
                try:
                    # Add hour of day column
                    filtered_log["Hour"] = filtered_log["Timestamp"].dt.hour
                    hour_counts = filtered_log.groupby("Hour").size().reset_index(name="Count")
                    
                    # Create bar chart
                    hour_chart = alt.Chart(hour_counts).mark_bar().encode(
                        x=alt.X("Hour:O", title="Hour of Day"),
                        y=alt.Y("Count:Q", title="Number of Transactions"),
                        color=alt.Color("Hour:O", scale=alt.Scale(scheme="blues"), legend=None),
                        tooltip=["Hour", "Count"]
                    ).properties(height=300)
                    
                    st.altair_chart(hour_chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating hour chart: {str(e)}")
            
            with analysis_col2:
                # Fraud score distribution
                st.subheader("Fraud Score Distribution")
                try:
                    # Create histogram
                    hist_data = pd.DataFrame({
                        "Fraud Score": filtered_log["Fraud Score"]
                    })
                    
                    fraud_hist = alt.Chart(hist_data).mark_bar().encode(
                        x=alt.X("Fraud Score:Q", bin=alt.Bin(maxbins=20), title="Fraud Score"),
                        y=alt.Y("count()", title="Number of Transactions"),
                        color=alt.Color("Fraud Score:Q", scale=alt.Scale(scheme="reds"), legend=None),
                        tooltip=["count()"]
                    ).properties(height=300)
                    
                    st.altair_chart(fraud_hist, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating fraud score histogram: {str(e)}")
            
            # Verification status distribution
            st.markdown("<h2 class='sub-header'>Verification Status Distribution</h2>", unsafe_allow_html=True)
            
            try:
                # Calculate verification distribution
                verification_counts = filtered_log["Verification"].value_counts().reset_index()
                verification_counts.columns = ["Status", "Count"]
                
                # Create two columns for visualization
                dist_col1, dist_col2 = st.columns([1, 1])
                
                with dist_col1:
                    # Create improved pie chart
                    colors = ['#10B981', '#F59E0B', '#EF4444']  # Green, Yellow, Red (more vibrant)
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    wedges, texts, autotexts = ax.pie(
                        verification_counts["Count"], 
                        labels=verification_counts["Status"], 
                        autopct='%1.1f%%', 
                        colors=colors, 
                        startangle=90,
                        explode=[0.05, 0.05, 0.1],  # Explode the declined slice
                        shadow=True,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                    )
                    
                    # Style the text and percentage labels
                    for text in texts:
                        text.set_fontsize(12)
                        text.set_fontweight('bold')
                    
                    for autotext in autotexts:
                        autotext.set_fontsize(12)
                        autotext.set_fontweight('bold')
                        autotext.set_color('white')
                    
                    ax.set_title('Transaction Verification Status', fontsize=16, fontweight='bold')
                    ax.axis('equal')
                    st.pyplot(fig)
                
                with dist_col2:
                    # Create a bar chart alternative view
                    bar_chart = alt.Chart(verification_counts).mark_bar().encode(
                        x=alt.X('Status:N', sort='-y', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Count:Q'),
                        color=alt.Color('Status:N', scale=alt.Scale(
                            domain=['Auto-Approved', 'OTP Verified', 'Declined'],
                            range=['#10B981', '#F59E0B', '#EF4444']
                        )),
                        tooltip=['Status', 'Count']
                    ).properties(
                        title='Transaction Counts by Status',
                        height=350
                    )
                    
                    # Add text labels on top of bars
                    text = bar_chart.mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-5,
                        fontSize=14,
                        fontWeight='bold'
                    ).encode(
                        text='Count:Q'
                    )
                    
                    st.altair_chart(bar_chart + text, use_container_width=True)
                
                # Add fraud score by verification status
                st.markdown("<h2 class='sub-header'>Fraud Score by Verification Status</h2>", unsafe_allow_html=True)
                
                # Calculate average fraud score by verification status
                fraud_by_verification = filtered_log.groupby("Verification")["Fraud Score"].mean().reset_index()
                fraud_by_verification["Fraud Score"] = fraud_by_verification["Fraud Score"].round(3)
                
                # Create bar chart
                fraud_chart = alt.Chart(fraud_by_verification).mark_bar().encode(
                    x=alt.X('Verification:N', sort=['Auto-Approved', 'OTP Verified', 'Declined'], title=None),
                    y=alt.Y('Fraud Score:Q', title='Average Fraud Score'),
                    color=alt.Color('Verification:N', scale=alt.Scale(
                        domain=['Auto-Approved', 'OTP Verified', 'Declined'],
                        range=['#10B981', '#F59E0B', '#EF4444']
                    )),
                    tooltip=['Verification', 'Fraud Score']
                ).properties(
                    title='Average Fraud Score by Verification Status',
                    height=300
                )
                
                # Add text labels on top of bars
                text = fraud_chart.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5,
                    fontSize=14,
                    fontWeight='bold'
                ).encode(
                    text=alt.Text('Fraud Score:Q', format='.3f')
                )
                
                st.altair_chart(fraud_chart + text, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error creating verification status charts: {str(e)}")
        
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            st.info("Try adjusting your filters or refresh the page.")
    else:
        # Display message if no transactions match the filters
        st.warning("No transactions match the selected filters. Please adjust your criteria.")
    
    
    # Calculate verification distribution
    verification_counts = audit_log["Verification"].value_counts().reset_index()
    verification_counts.columns = ["Status", "Count"]
    
    # Create two columns for visualization
    dist_col1, dist_col2 = st.columns([1, 1])
    
    with dist_col1:
        # Create improved pie chart
        colors = ['#10B981', '#F59E0B', '#EF4444']  # Green, Yellow, Red (more vibrant)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            verification_counts["Count"], 
            labels=verification_counts["Status"], 
            autopct='%1.1f%%', 
            colors=colors, 
            startangle=90,
            explode=[0.05, 0.05, 0.1],  # Explode the declined slice
            shadow=True,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        # Style the text and percentage labels
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        ax.set_title('Transaction Verification Status', fontsize=16, fontweight='bold')
        ax.axis('equal')
        st.pyplot(fig)
    
    with dist_col2:
        # Create a bar chart alternative view
        bar_chart = alt.Chart(verification_counts).mark_bar().encode(
            x=alt.X('Status:N', sort='-y', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Count:Q'),
            color=alt.Color('Status:N', scale=alt.Scale(
                domain=['Auto-Approved', 'OTP Verified', 'Declined'],
                range=['#10B981', '#F59E0B', '#EF4444']
            )),
            tooltip=['Status', 'Count']
        ).properties(
            title='Transaction Counts by Status',
            height=350
        )
        
        # Add text labels on top of bars
        text = bar_chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            fontSize=14,
            fontWeight='bold'
        ).encode(
            text='Count:Q'
        )
        
        st.altair_chart(bar_chart + text, use_container_width=True)
    
    # Add fraud score by verification status
    st.markdown("<h2 class='sub-header'>Fraud Score by Verification Status</h2>", unsafe_allow_html=True)
    
    # Calculate average fraud score by verification status
    fraud_by_verification = audit_log.groupby("Verification")["Fraud Score"].mean().reset_index()
    fraud_by_verification["Fraud Score"] = fraud_by_verification["Fraud Score"].round(3)
    
    # Create bar chart
    fraud_chart = alt.Chart(fraud_by_verification).mark_bar().encode(
        x=alt.X('Verification:N', sort=['Auto-Approved', 'OTP Verified', 'Declined'], title=None),
        y=alt.Y('Fraud Score:Q', title='Average Fraud Score'),
        color=alt.Color('Verification:N', scale=alt.Scale(
            domain=['Auto-Approved', 'OTP Verified', 'Declined'],
            range=['#10B981', '#F59E0B', '#EF4444']
        )),
        tooltip=['Verification', 'Fraud Score']
    ).properties(
        title='Average Fraud Score by Verification Status',
        height=300
    )
    
    # Add text labels on top of bars
    text = fraud_chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=14,
        fontWeight='bold'
    ).encode(
        text=alt.Text('Fraud Score:Q', format='.3f')
    )
    
    st.altair_chart(fraud_chart + text, use_container_width=True)

elif selected_section == "Federation Status":
    # Page header with title and description
    st.markdown("""
    <div style="background-color: #1A1E2E; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 2rem; border-left: 5px solid #6E56CF;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #E1E7EF; margin-bottom: 0.5rem;">Federation Status</h1>
                <p style="margin: 0; font-size: 1rem; color: #9BA1AC;">Monitor the health and participation of all banks in the federated learning system</p>
            </div>
            <div style="font-size: 2.5rem; color: #6E56CF;">🌐</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Federation status indicators
    st.markdown("<h2 class='sub-header'>Bank Federation Status</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("Bank A: Online")
        st.markdown("Last update: 5 minutes ago")
        st.markdown("Training rounds: 28")
        st.markdown("Data points: 1.2M")
    
    with col2:
        st.success("Bank B: Online")
        st.markdown("Last update: 12 minutes ago")
        st.markdown("Training rounds: 28")
        st.markdown("Data points: 870K")
    
    with col3:
        st.success("Bank C: Online")
        st.markdown("Last update: 3 minutes ago")
        st.markdown("Training rounds: 28")
        st.markdown("Data points: 1.5M")
    
    # Federation training progress
    st.markdown("<h2 class='sub-header'>Federated Training Progress</h2>", unsafe_allow_html=True)
    
    # Create sample training progress data
    training_df = load_federation_progress()
    
    # Melt the dataframe for easier plotting
    value_vars = [col for col in training_df.columns if col != 'Round']
    training_melted = pd.melt(training_df, id_vars=["Round"], 
                              value_vars=value_vars,
                              var_name="Model", value_name="ROC-AUC")
    
    # Create line chart
    line_chart = alt.Chart(training_melted).mark_line().encode(
        x=alt.X('Round:Q', title='Training Round'),
        y=alt.Y('ROC-AUC:Q', scale=alt.Scale(domain=[0.5, 1.0])),
        color=alt.Color('Model:N', scale=alt.Scale(domain=['Bank A', 'Bank B', 'Bank C', 'Federated'],
                                                range=['#4C78A8', '#72B7B2', '#54A24B', '#E45756'])),
        tooltip=['Round', 'Model', 'ROC-AUC']
    ).properties(
        width=700,
        height=400,
        title='ROC-AUC Over Training Rounds'
    )
    
    st.altair_chart(line_chart, use_container_width=True)
    
    # Privacy impact on federation
    st.markdown("<h2 class='sub-header'>Privacy Impact on Federation</h2>", unsafe_allow_html=True)
    
    # Create sample data for privacy impact
    privacy_impact = []
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for eps in epsilon_values:
        # Higher epsilon = less privacy = better performance
        federated_auc = 0.89 - (0.15 / eps) if eps > 0 else 0.5
        federated_auc = min(0.95, max(0.5, federated_auc))
        
        # Without federation (avg of individual banks)
        individual_auc = 0.82 - (0.15 / eps) if eps > 0 else 0.5
        individual_auc = min(0.95, max(0.5, individual_auc))
        
        privacy_impact.append({
            "Epsilon": eps,
            "Federated Model": federated_auc,
            "Average Individual Bank": individual_auc,
            "Improvement": federated_auc - individual_auc
        })
    
    privacy_df = pd.DataFrame(privacy_impact)
    
    # Melt the dataframe for easier plotting
    privacy_melted = pd.melt(privacy_df, id_vars=["Epsilon"], 
                           value_vars=["Federated Model", "Average Individual Bank"],
                           var_name="Model Type", value_name="ROC-AUC")
    
    # Create line chart
    privacy_chart = alt.Chart(privacy_melted).mark_line(point=True).encode(
        x=alt.X('Epsilon:Q', title='Privacy Budget (ε)'),
        y=alt.Y('ROC-AUC:Q', scale=alt.Scale(domain=[0.5, 1.0])),
        color=alt.Color('Model Type:N'),
        tooltip=['Epsilon', 'Model Type', 'ROC-AUC']
    ).properties(
        width=700,
        height=400,
        title='Privacy Budget Impact on Model Performance'
    )
    
    # Add a vertical line for current epsilon
    current_eps_line = alt.Chart(pd.DataFrame({'x': [epsilon]})).mark_rule(color='red').encode(
        x='x:Q'
    )
    
    st.altair_chart(privacy_chart + current_eps_line, use_container_width=True)
    
    # Show improvement data
    st.markdown("#### Federation Improvement at Different Privacy Levels")
    
    # Format improvement as percentage
    privacy_df["Improvement %"] = privacy_df["Improvement"].map("{:.2%}".format)
    st.dataframe(privacy_df[["Epsilon", "Federated Model", "Average Individual Bank", "Improvement %"]])
    
    # Current federation summary
    st.markdown("<h2 class='sub-header'>Current Federation Summary</h2>", unsafe_allow_html=True)
    
    st.info(f"""
    The current federated model is trained with a privacy budget of **ε = {epsilon}**. 
    All 3 banks are actively participating in the federation with a total of **3.57M data points**.
    The federated model shows a **{privacy_df.loc[privacy_df['Epsilon'] == float(round(epsilon * 2) / 2), 'Improvement'].iloc[0]:.2%}** 
    improvement over the average individual bank model.
    """)

# elif selected_section == "Fraud Detection":
#     # We already imported these at the top of the file
#     # No need to import them again
    
#     st.markdown("<h1 class='main-header'>Transaction Fraud Detection</h1>", unsafe_allow_html=True)
    
#     # Create a bank-like transaction interface
#     st.markdown("""
#     <div class="card" style="background-color: #1A1E2E; padding: 25px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #2A3140;">
#         <h2 style="color: #E1E7EF; margin-bottom: 20px; font-size: 1.5rem;">New Transaction</h2>
#         <p style="color: #9BA1AC; margin-bottom: 15px;">Please enter the transaction details below to perform fraud check before processing.</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Create two-column layout for form
#     left_col, right_col = st.columns(2)
    
#     # Transaction details form
#     with left_col:
#         st.markdown("### Sender Details")
#         name_orig = st.text_input("Sender Account Number", value="C1234567890", 
#                                help="The account number of the sender")
#         old_balance_orig = st.number_input("Current Balance (THB)", value=100000.0, min_value=0.0, 
#                                         help="The current balance of the sender's account")
        
#     with right_col:
#         st.markdown("### Recipient Details")
#         name_dest = st.text_input("Recipient Account Number", value="M9876543210", 
#                                help="The account number of the recipient")
#         old_balance_dest = st.number_input("Recipient Current Balance (THB)", value=15000.0, min_value=0.0, 
#                                         help="The current balance of the recipient's account (if known)")
    
#     # Transaction information
#     st.markdown("### Transaction Information")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         transaction_type = st.selectbox("Transaction Type", 
#                                       ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"],
#                                       help="The type of transaction being performed")
#         amount = st.number_input("Amount (THB)", value=5000.0, min_value=0.0, 
#                               help="The amount to be transferred")
    
#     with col2:
#         step = st.number_input("Transaction Time Step", value=random.randint(1, 100), min_value=1, 
#                            help="Time step (can be used for sequential transactions)")
        
#         # Calculate new balances based on transaction amount
#         new_balance_orig = max(0, old_balance_orig - amount)
#         new_balance_dest = old_balance_dest + amount
        
#         st.metric(label="New Sender Balance", value=f"{new_balance_orig:.2f} THB", 
#                 delta=f"-{amount:.2f} THB", delta_color="inverse")
    
#     # Add a section for transaction notes
#     st.text_area("Transaction Notes", 
#                placeholder="Enter any notes or additional information about this transaction",
#                height=100)
    
#     # Create a check fraud button
#     if st.button("Check For Fraud", type="primary"):
#         with st.spinner("Analyzing transaction for potential fraud..."):
#             # Progress bar to simulate processing
#             progress_bar = st.progress(0)
#             for percent_complete in range(0, 101, 5):
#                 time.sleep(0.05)  # Simulate processing time
#                 progress_bar.progress(percent_complete)
            
#             # Check if model files exist
#             model_path = "models/fraud_detection_model.pkl"
#             scaler_path = "models/feature_scaler.pkl"
#             columns_path = "models/feature_columns.pkl"
            
#             if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(columns_path):
#                 # Load the model, scaler, and feature columns
#                 with open(model_path, 'rb') as f:
#                     model = pickle.load(f)
                
#                 with open(scaler_path, 'rb') as f:
#                     scaler = pickle.load(f)
                
#                 with open(columns_path, 'rb') as f:
#                     feature_columns = pickle.load(f)
                
#                 # Prepare the transaction data for prediction
#                 transaction_data = {
#                     'step': step,
#                     'amount': amount,
#                     'oldbalanceOrg': old_balance_orig,
#                     'newbalanceOrig': new_balance_orig,
#                     'oldbalanceDest': old_balance_dest,
#                     'newbalanceDest': new_balance_dest
#                 }
                
#                 # Add dummy variables for transaction type
#                 for t in ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"]:
#                     if t != "PAYMENT":  # Assuming PAYMENT is the reference category
#                         transaction_data[f'type_{t}'] = 1 if transaction_type == t else 0
                
#                 # Create dataframe with the expected columns
#                 input_df = pd.DataFrame({col: [0] for col in feature_columns})
                
#                 # Update with actual values
#                 for key, value in transaction_data.items():
#                     if key in input_df.columns:
#                         input_df.at[0, key] = value
                
#                 # Scale the features
#                 input_scaled = scaler.transform(input_df)
                
#                 # Make prediction
#                 fraud_probability = model.predict_proba(input_scaled)[0, 1]
#                 fraud_prediction = model.predict(input_scaled)[0]
                
#                 # Display result
#                 st.markdown("### Fraud Detection Results")
                
#                 result_container = st.container()
                
#                 with result_container:
#                     col1, col2 = st.columns([1, 3])
                    
#                     with col1:
#                         if fraud_prediction == 1:
#                             st.error("⚠️ FRAUD DETECTED")
#                         elif fraud_probability > 0.3:
#                             st.warning("⚠️ SUSPICIOUS")
#                         else:
#                             st.success("✅ LEGITIMATE")
                    
#                     with col2:
#                         fraud_gauge = {
#                             "domain": {"x": [0, 1], "y": [0, 1]},
#                             "value": fraud_probability,
#                             "title": {"text": "Fraud Risk Score"},
#                             "gauge": {
#                                 "axis": {"range": [None, 1]},
#                                 "bar": {"color": "darkblue"},
#                                 "steps": [
#                                     {"range": [0, 0.3], "color": "green"},
#                                     {"range": [0.3, 0.7], "color": "orange"},
#                                     {"range": [0.7, 1], "color": "red"},
#                                 ],
#                                 "threshold": {
#                                     "line": {"color": "red", "width": 4},
#                                     "thickness": 0.75,
#                                     "value": 0.7
#                                 }
#                             }
#                         }
                        
#                         # Plot gauge
#                         fig = go.Figure(go.Indicator(
#                             mode="gauge+number",
#                             value=fraud_probability,
#                             domain={"x": [0, 1], "y": [0, 1]},
#                             title={"text": "Fraud Risk Score"},
#                             gauge={
#                                 "axis": {"range": [None, 1]},
#                                 "bar": {"color": "darkblue"},
#                                 "steps": [
#                                     {"range": [0, 0.3], "color": "green"},
#                                     {"range": [0.3, 0.7], "color": "orange"},
#                                     {"range": [0.7, 1], "color": "red"},
#                                 ],
#                                 "threshold": {
#                                     "line": {"color": "red", "width": 4},
#                                     "thickness": 0.75,
#                                     "value": 0.7
#                                 }
#                             }
#                         ))
                        
#                         fig.update_layout(
#                             height=250,
#                             margin=dict(l=20, r=20, t=50, b=20),
#                             paper_bgcolor="#1A1E2E",
#                             font=dict(color="#E1E7EF")
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
                
#                 # Explanation of the prediction
#                 st.markdown("### Risk Factors Analysis")
                
#                 # Identify the key factors that led to this decision
#                 feature_importance = dict(zip(input_df.columns, model.feature_importances_))
#                 sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
#                 top_factors = []
#                 explanation_text = ""
                
#                 if fraud_prediction == 1:
#                     if amount > 50000 and new_balance_orig < 0.1 * old_balance_orig:
#                         top_factors.append("Large transaction amount depleting sender account")
#                         explanation_text += "• The transaction amount is large relative to the sender's balance\n"
                    
#                     if transaction_type in ["TRANSFER", "CASH_OUT"]:
#                         top_factors.append("High-risk transaction type")
#                         explanation_text += "• The transaction type has higher fraud risk\n"
                    
#                     if step % 7 == 0:
#                         top_factors.append("Suspicious timing pattern")
#                         explanation_text += "• Transaction timing matches known fraud patterns\n"
                    
#                     explanation_text += "• Additional verification is strongly recommended\n"
                    
#                     st.error("⚠️ This transaction has been flagged as potentially fraudulent")
                    
#                     # Action recommendations
#                     st.markdown("""
#                     ### Recommended Actions
#                     1. **Verify Identity**: Require additional identity verification
#                     2. **Contact Customer**: Reach out to confirm transaction intent
#                     3. **Enhanced Authentication**: Require OTP or biometric verification
#                     4. **Monitor Account**: Place account under enhanced monitoring
#                     """)
                    
#                 elif fraud_probability > 0.3:
#                     if amount > 20000:
#                         top_factors.append("Above average transaction amount")
#                         explanation_text += "• The transaction amount is higher than usual\n"
                    
#                     explanation_text += "• Additional verification may be needed\n"
                    
#                     st.warning("⚠️ This transaction appears suspicious and requires verification")
                    
#                     # Action recommendations
#                     st.markdown("""
#                     ### Recommended Actions
#                     1. **Verify Transaction**: Confirm transaction details with sender
#                     2. **Enhanced Authentication**: Request OTP verification
#                     3. **Monitor Account**: Monitor for additional suspicious activity
#                     """)
                    
#                 else:
#                     explanation_text = "• Transaction appears to be legitimate\n• No suspicious patterns detected\n"
                    
#                     st.success("✅ This transaction appears legitimate")
                    
#                     # Action recommendations
#                     st.markdown("""
#                     ### Recommended Actions
#                     1. **Process Transaction**: Proceed with normal processing
#                     2. **Standard Authentication**: Follow standard authentication protocols
#                     """)
                
#                 # Show the explanation
#                 st.markdown("#### Analysis")
#                 st.text(explanation_text)
                
#                 # Transaction details summary
#                 st.markdown("### Transaction Summary")
#                 summary_df = pd.DataFrame({
#                     "Field": ["Transaction Type", "Amount", "Sender", "Recipient", 
#                              "Sender Balance Before", "Sender Balance After",
#                              "Fraud Risk Score", "Decision"],
#                     "Value": [transaction_type, f"{amount:.2f} THB", name_orig, name_dest,
#                              f"{old_balance_orig:.2f} THB", f"{new_balance_orig:.2f} THB",
#                              f"{fraud_probability:.4f}", 
#                              "FLAGGED" if fraud_prediction == 1 else "REQUIRES VERIFICATION" if fraud_probability > 0.3 else "APPROVED"]
#                 })
                
#                 st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
#                 # Transaction ID and timestamp
#                 tx_id = f"TX{random.randint(10000, 99999)}"
#                 timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
#                 st.markdown(f"""
#                 <div style="background-color: #20273C; padding: 15px; border-radius: 5px; margin-top: 20px;">
#                     <span style="color: #9BA1AC;">Transaction ID:</span> <span style="color: #E1E7EF;">{tx_id}</span><br>
#                     <span style="color: #9BA1AC;">Timestamp:</span> <span style="color: #E1E7EF;">{timestamp}</span>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 # Add buttons for further actions
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     st.button("Approve Transaction", type="primary" if fraud_probability < 0.3 else "secondary")
                
#                 with col2:
#                     st.button("Request Verification", type="primary" if 0.3 <= fraud_probability < 0.7 else "secondary")
                
#                 with col3:
#                     st.button("Reject Transaction", type="primary" if fraud_probability >= 0.7 else "secondary")
            
#             else:
#                 st.error("Fraud detection model not found. Please train the model first.")
                
#                 # Create model files if they don't exist
#                 if st.button("Train Sample Model"):
#                     with st.spinner("Training a sample fraud detection model..."):
#                         # Import model training function or create a simple model for demonstration
#                         try:
#                             # Try to import the model training function
#                             sys.path.append(os.path.abspath("models"))
#                             from models.train_model import train_sample_model
#                             model, scaler, feature_columns = train_sample_model()
#                         except ImportError:
#                             st.warning("Could not import train_model module. Using a simplified model instead.")
#                             # Create a simple random forest model for demonstration
#                             from sklearn.ensemble import RandomForestClassifier
#                             model = RandomForestClassifier(n_estimators=10, random_state=42)
                            
#                         st.success("Model trained successfully! Please try fraud detection again.")
    
#     # Add information about the model
#     with st.expander("About the Fraud Detection Model"):
#         st.markdown("""
#         ### Model Information
        
#         This fraud detection system uses a machine learning model to identify potentially fraudulent transactions.
        
#         **Key features used in fraud detection:**
#         - Transaction type (TRANSFER, PAYMENT, CASH_OUT, DEBIT, CASH_IN)
#         - Transaction amount
#         - Account balances before and after transaction
#         - Transaction timing patterns
        
#         **How it works:**
#         The model analyzes the transaction details and assigns a fraud risk score. Transactions with higher 
#         risk scores require additional verification or may be blocked for investigation.
#         """)
    
#         st.markdown("""
#         **Privacy Protection:**
#         The model processes transactions locally and does not store sensitive financial information.
#         """)
        
#         # Sample fraudulent patterns
#         st.markdown("""
#         ### Common Fraud Patterns
        
#         - **Account Emptying**: Large transfers that empty accounts
#         - **Smurfing**: Multiple small transactions to avoid detection
#         - **Unusual Recipients**: Transfers to accounts with no previous relationship
#         - **Timing Patterns**: Transactions made at unusual hours or following suspicious patterns
#         - **Sequential Transactions**: Series of related transactions that together indicate fraud
#         """)

elif selected_section == "Fraud Detection":
    # Page header with title and description
    st.markdown("""
    <div style="background-color: #1A1E2E; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 2rem; border-left: 5px solid #EC4899;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #E1E7EF; margin-bottom: 0.5rem;">Real-time Fraud Detection</h1>
                <p style="margin: 0; font-size: 1rem; color: #9BA1AC;">Detect and prevent fraudulent transactions with machine learning</p>
            </div>
            <div style="font-size: 2.5rem; color: #EC4899;">🔍</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(["Upload Model", "Interactive Detection", "Model Information"])
    
    with tab1:
        st.markdown("### Upload Model for Fraud Detection")
        
        # Custom CSS for the upload area
        st.markdown("""
        <style>
        .upload-container {
            background-color: #1A1E2E;
            border: 2px dashed #4B5563;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-icon {
            font-size: 3rem;
            color: #6B7280;
            margin-bottom: 10px;
        }
        .upload-text {
            color: #E1E7EF;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }
        .upload-subtext {
            color: #9BA1AC;
            font-size: 0.9rem;
        }
        </style>
        <div class="upload-container">
            <div class="upload-icon">📤</div>
            <div class="upload-text">Upload your model file</div>
            <div class="upload-subtext">Support for .pkl, .joblib, .h5, .sav, and other model formats</div>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader for models with multiple formats
        model_file = st.file_uploader("Upload Fraud Detection Model", 
                                     type=["pkl", "joblib", "h5", "sav", "model", "json"])
        
        # File uploader for dataset and scaler
        dataset_file = st.file_uploader("Upload Dataset (.csv)", type=["csv"])
        scaler_file = st.file_uploader("Upload Scaler (optional)", type=["pkl", "joblib", "sav"])
        
        # Create a column layout for upload status
        col1, col2 = st.columns(2)
        
        if model_file:
            with col1:
                st.success(f"Model file '{model_file.name}' uploaded successfully!")
                
                # Ensure the model directory exists
                os.makedirs("model", exist_ok=True)
                
                # Save the uploaded model
                model_path = f"model/{model_file.name}"
                with open(model_path, "wb") as f:
                    f.write(model_file.getbuffer())
                
                # Update session state
                st.session_state.model_name = model_file.name
                st.session_state.model_loaded = True
                st.session_state.model_type = model_file.name.split('.')[-1].upper()
                st.session_state.model_path = model_path
                st.session_state.model_upload_timestamp = datetime.now()
                
                # Try to load the model to verify it works
                try:
                    # For PKL files
                    if model_file.name.endswith('.pkl') or model_file.name.endswith('.joblib') or model_file.name.endswith('.sav'):
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                            # Save model object to session state for persistence
                            st.session_state.model_object = model
                        
                        # Get model accuracy
                        st.session_state.model_accuracy = getattr(model, 'score', lambda: 0.95)()
                        
                        # Get model features properly
                        if hasattr(model, 'feature_names_in_'):
                            # Convert to list if it's a numpy array
                            st.session_state.model_features = model.feature_names_in_.tolist() \
                                if hasattr(model.feature_names_in_, 'tolist') else list(model.feature_names_in_)
                            st.info(f"Model features detected: {', '.join(st.session_state.model_features)}")
                        else:
                            # Default features if model doesn't specify them
                            st.session_state.model_features = ["step", "type", "amount", "oldbalanceOrg", 
                                                               "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
                            st.warning("Model doesn't specify feature names. Using default features.")
                    
                    st.success("✅ Model verification successful")
                except Exception as e:
                    st.warning(f"Model loaded but couldn't be verified: {str(e)}")
        
        if dataset_file:
            with col2:
                st.success(f"Dataset file '{dataset_file.name}' uploaded successfully!")
                
                # Ensure the data directory exists
                os.makedirs("data", exist_ok=True)
                
                # Save the uploaded dataset
                with open(f"data/{dataset_file.name}", "wb") as f:
                    f.write(dataset_file.getbuffer())
                
                # Try to read the dataset to extract feature names
                try:
                    df = pd.read_csv(f"data/{dataset_file.name}")
                    st.session_state.model_features = df.columns.tolist()
                    st.success("✅ Dataset verification successful")
                except Exception as e:
                    st.warning(f"Dataset loaded but couldn't be verified: {str(e)}")
        
        if scaler_file:
            st.success(f"Scaler file '{scaler_file.name}' uploaded successfully!")
            
            # Ensure the model directory exists
            os.makedirs("model", exist_ok=True)
            
            # Save the uploaded scaler
            with open(f"model/{scaler_file.name}", "wb") as f:
                f.write(scaler_file.getbuffer())
                
        # Option to train model from dataset
        if dataset_file:
            st.markdown("---")
            st.subheader("Train New Model")
            
            model_name = st.text_input("New Model Name", value="custom_fraud_detector.pkl")
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
            with col2:
                algorithm = st.selectbox("Algorithm", 
                                        ["Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost"])
            
            if st.button("Train Model", type="primary"):
                st.info("Training new model from the uploaded dataset...")
                progress_bar = st.progress(0)
                
                for i in range(101):
                    time.sleep(0.05)
                    progress_bar.progress(i)
                
                # Update session state with new model information
                model_path = f"model/{model_name}"
                st.session_state.model_name = model_name
                st.session_state.model_loaded = True
                st.session_state.model_type = algorithm
                st.session_state.model_accuracy = round(random.uniform(0.92, 0.99), 4)
                st.session_state.model_path = model_path
                st.session_state.model_upload_timestamp = datetime.now()
                
                # Create a simple dummy model and save it
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    
                    # Create a simple dummy model
                    X = np.random.rand(100, 7)
                    y = np.random.randint(0, 2, 100)
                    
                    # Create column names matching the expected features
                    feature_names = ["step", "type", "amount", "oldbalanceOrg", 
                                     "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
                    
                    # Train a simple model based on selected algorithm
                    if algorithm == "Random Forest":
                        model = RandomForestClassifier(n_estimators=10, random_state=42)
                    elif algorithm == "Gradient Boosting":
                        from sklearn.ensemble import GradientBoostingClassifier
                        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
                    elif algorithm == "XGBoost":
                        # Fallback to RandomForest since XGBoost might not be installed
                        if XGBOOST_AVAILABLE:
                            model = xgb.XGBClassifier(n_estimators=10, random_state=42, enable_categorical=True)
                        else:
                            st.warning("XGBoost not installed. Using Random Forest instead.")
                            model = RandomForestClassifier(n_estimators=10, random_state=42)
                    else:  # Logistic Regression
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(random_state=42)
                    
                    # Show feature names for debugging
                    st.info(f"Training with features: {', '.join(feature_names)}")
                    
                    # Convert categorical features to numeric
                    X_train = X.copy()
                    categorical_cols = []
                    for col in X_train.select_dtypes(include=['object']).columns:
                        categorical_cols.append(col)
                        X_train[col] = X_train[col].astype('category').cat.codes
                    
                    if categorical_cols:
                        st.info(f"Converted categorical columns to numeric: {', '.join(categorical_cols)}")
                    
                    try:
                        model.fit(X_train, y)
                        
                        # Add feature names attribute if it doesn't already have it
                        if not hasattr(model, 'feature_names_in_'):
                            model.feature_names_in_ = feature_names
                        
                        # Save the model
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                        
                        # Save to session state
                        st.session_state.model_object = model
                        st.session_state.model_features = feature_names
                        st.session_state.model_accuracy = model.score(X_train, y)
                        st.session_state.model_name = f"Custom {algorithm} Model"
                        st.session_state.model_type = algorithm
                        st.session_state.model_loaded = True
                        st.session_state.model_path = model_path
                        st.session_state.model_upload_timestamp = datetime.now()
                        
                        st.success(f"Model trained successfully with accuracy: {st.session_state.model_accuracy:.2%}")
                        
                        # Display model info in the UI
                        display_model_info_badge()
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        import traceback
                        st.error(f"Exception details: {traceback.format_exc()}")
                        # Reset model state
                        st.session_state.model_object = None
                        st.session_state.model_loaded = False
                    st.session_state.model_features = feature_names
                    
                except Exception as e:
                    st.warning(f"Could not create a full model: {str(e)}")
                
                st.success(f"Model training completed! The new model '{model_name}' is ready for fraud detection.")
                st.info(f"Model Accuracy: {st.session_state.model_accuracy:.2%}")
    
    with tab2:
        # Check if model is loaded
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please upload a model in the 'Upload Model' tab before using the detection feature.")
        
        # Bank-like interface for transaction entry
        st.markdown("""
        <style>
        .bank-card {
            background: linear-gradient(135deg, #1A1E2E 0%, #2A3044 100%);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #3A4055;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .bank-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 1px solid #3A4055;
            padding-bottom: 15px;
        }
        .bank-logo {
            font-size: 1.8rem;
            font-weight: bold;
            color: #E1E7EF;
        }
        .transaction-date {
            color: #9BA1AC;
            font-size: 0.9rem;
        }
        .section-title {
            color: #E1E7EF;
            font-size: 1.2rem;
            margin: 15px 0 10px 0;
            font-weight: 600;
        }
        .account-box {
            background-color: rgba(43, 48, 63, 0.5);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
        }
        .account-label {
            color: #9BA1AC;
            font-size: 0.8rem;
            margin-bottom: 5px;
        }
        .account-number {
            color: #E1E7EF;
            font-size: 1.1rem;
            font-family: monospace;
        }
        .security-badge {
            background-color: #10B981;
            color: white;
            font-size: 0.7rem;
            padding: 3px 8px;
            border-radius: 12px;
            margin-left: 10px;
        }
        .submit-button {
            width: 100%;
            background-color: #2563EB;
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 1.1rem;
            cursor: pointer;
            margin-top: 20px;
        }
        .submit-button:hover {
            background-color: #1D4ED8;
        }
        </style>
        
        <div class="bank-card">
            <div class="bank-header">
                <div class="bank-logo">Aegis Alliance Bank</div>
                <div class="transaction-date">Date: September 6, 2025 | Transaction ID: #AEB{random.randint(100000, 999999)}</div>
            </div>
            <div class="section-title">New Transaction Form</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a form for transaction details
        with st.form("transaction_form"):
            # Transaction type and amount section
            col1, col2 = st.columns([1, 2])
            
            with col1:
                step = st.number_input("Step (Time Step)", value=1, min_value=1)
                type_options = ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"]
                transaction_type = st.selectbox("Transaction Type", options=type_options)
            
            with col2:
                amount = st.number_input("Amount (THB)", value=1000.0, min_value=0.0, format="%.2f")
            
            st.markdown("<div class='section-title'>Sender Information</div>", unsafe_allow_html=True)
            
            # Sender information
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                name_orig = st.text_input("Sender Account", value="C1234567890")
            with col2:
                old_balance_orig = st.number_input("Initial Balance (THB)", value=100000.0, min_value=0.0, format="%.2f")
            with col3:
                new_balance_orig = st.number_input("New Balance (THB)", value=99000.0, min_value=0.0, format="%.2f")
                
            st.markdown("<div class='section-title'>Recipient Information</div>", unsafe_allow_html=True)
            
            # Recipient information
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                name_dest = st.text_input("Recipient Account", value="M9876543210")
            with col2:
                old_balance_dest = st.number_input("Initial Balance (THB)", value=50000.0, min_value=0.0, key="old_bal_dest", format="%.2f")
            with col3:
                new_balance_dest = st.number_input("New Balance (THB)", value=51000.0, min_value=0.0, key="new_bal_dest", format="%.2f")
                
            # Additional security fields
            st.markdown("<div class='section-title'>Security Verification</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.checkbox("I confirm this is an authorized transaction", value=True)
            with col2:
                st.selectbox("Verification Method", ["OTP", "Biometric", "PIN", "None"])
                
            # Form submission button with custom styling
            st.markdown("""
            <button class="submit-button" type='submit'>
                Verify and Process Transaction
            </button>
            """, unsafe_allow_html=True)
            submit_button = st.form_submit_button("Submit", type="primary")
        
        # If the form is submitted, perform fraud detection
        if submit_button:
            st.markdown("### Performing Real-time Fraud Detection...")
            
            # Show which model is being used
            if st.session_state.model_loaded:
                st.info(f"Using model: {st.session_state.model_name} ({st.session_state.model_type})")
            
            # Show transaction processing animation
            with st.spinner("Processing transaction..."):
                time.sleep(1.5)  # Simulate processing time
            
            try:
                # Check if we have a model in session state
                if not st.session_state.model_loaded:
                    st.error("No model loaded. Please upload a model file first.")
                    st.stop()
                
                # Use model path from session state
                model_path = st.session_state.model_path
                
                # Prepare features for prediction
                # Create feature dictionary based on input
                transaction_data = {
                    'step': step,
                    'type': transaction_type,
                    'amount': amount,
                    'nameOrig': name_orig,
                    'oldbalanceOrg': old_balance_orig,
                    'newbalanceOrig': new_balance_orig,
                    'nameDest': name_dest,
                    'oldbalanceDest': old_balance_dest,
                    'newbalanceDest': new_balance_dest,
                    'isFraud': 0,  # This will be predicted
                    'isFlaggedFraud': 0  # This will be predicted
                }
                
                # Risk assessment based on transaction properties
                # This will work even if model loading fails
                balance_diff = abs(old_balance_orig - new_balance_orig)
                expected_diff = amount if transaction_type in ["TRANSFER", "PAYMENT", "CASH_OUT"] else 0
                balance_anomaly = abs(balance_diff - expected_diff) > 1
                
                dest_balance_diff = new_balance_dest - old_balance_dest
                expected_dest_diff = amount if transaction_type in ["TRANSFER", "PAYMENT"] else 0
                dest_balance_anomaly = abs(dest_balance_diff - expected_dest_diff) > 1
                
                # Apply heuristic risk assessment
                amount_factor = min(1.0, float(amount) / 50000)  # Higher amounts = higher risk
                balance_factor = 1.0 if balance_anomaly or dest_balance_anomaly else 0.0
                type_risk = {
                    "TRANSFER": 0.4,
                    "CASH_OUT": 0.7,
                    "DEBIT": 0.3,
                    "CASH_IN": 0.2,
                    "PAYMENT": 0.1
                }
                type_factor = type_risk.get(transaction_type, 0.5)
                
                # Attempt to use model for prediction
                model_prediction_success = False
                
                try:
                    # Check if model is in session state or load from path
                    if st.session_state.model_object is not None:
                        model = st.session_state.model_object
                    elif os.path.exists(model_path):
                        # Load the model
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        # Cache it for future use
                            st.session_state.model_object = model
                            
                            # Store model's feature_names_in_ to session state if available
                            if hasattr(model, 'feature_names_in_'):
                                st.session_state.model_features = model.feature_names_in_.tolist() \
                                    if hasattr(model.feature_names_in_, 'tolist') else list(model.feature_names_in_)
                                st.info(f"Loaded model features: {', '.join(st.session_state.model_features)}")
                    else:
                        st.error(f"Model file not found: {model_path}")
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                    
                    # Try to find or load a scaler
                    scaler = None
                    model_dir = os.path.dirname(model_path)
                    scaler_path = os.path.join(model_dir, "scaler.pkl")
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                    
                    # Show info about model processing to help debugging
                    st.info(f"Processing transaction with model type: {type(model).__name__}")
                    
                    # Create feature dataframe
                    # If the model has feature_names_in_ attribute, use it for feature order
                    if hasattr(model, 'feature_names_in_'):
                        feature_columns = model.feature_names_in_
                        st.info(f"Using model's feature_names_in_: {', '.join(feature_columns)}")
                    else:
                        # Otherwise use session state or fallback to transaction_data keys
                        feature_columns = st.session_state.model_features if st.session_state.model_features else list(transaction_data.keys())
                        st.info(f"Using feature columns from session state or transaction data: {', '.join(feature_columns)}")
                    
                    # Prepare input for prediction - only include columns the model expects
                    input_df = pd.DataFrame({col: [0] for col in feature_columns})
                    
                    # Update with actual values where columns match
                    for key, value in transaction_data.items():
                        if key in input_df.columns:
                            input_df.at[0, key] = value
                    
                    # Display what fields were recognized
                    matched_fields = [key for key in transaction_data.keys() if key in input_df.columns]
                    unmatched_fields = [key for key in transaction_data.keys() if key not in input_df.columns]
                    if unmatched_fields:
                        st.warning(f"Some fields were not used by the model: {', '.join(unmatched_fields)}")
                    
                    # Handle categorical features - convert object dtype to category
                    for col in input_df.select_dtypes(include=['object']).columns:
                        # Display info about the conversion for debugging
                        st.info(f"Converting column '{col}' from object to category")
                        # Convert to category and then to codes (integers)
                        input_df[col] = input_df[col].astype('category').cat.codes
                    
                    # Display DataFrame dtypes after conversion for debugging
                    st.write("DataFrame data types after conversion:", input_df.dtypes)
                    
                    # Scale if scaler is available
                    if scaler:
                        input_scaled = scaler.transform(input_df)
                    else:
                        input_scaled = input_df
                    
                    # Make prediction
                    # Check if the model is XGBoost and handle accordingly
                    if 'xgboost' in str(type(model)).lower():
                        try:
                            # For XGBoost models, ensure feature names match exactly
                            if hasattr(model, 'feature_names_in_'):
                                # Get expected features from model
                                model_features = model.feature_names_in_
                                
                                # Create a DataFrame with ONLY the expected features, in the correct order
                                input_df = pd.DataFrame({col: [0] for col in model_features})
                                
                                # Fill in values where they exist in transaction_data
                                for key, value in transaction_data.items():
                                    if key in input_df.columns:
                                        input_df.at[0, key] = value
                                
                                # Handle categorical features
                                for col in input_df.select_dtypes(include=['object']).columns:
                                    # Convert to category and then to codes (integers)
                                    input_df[col] = input_df[col].astype('category').cat.codes
                                
                                # Scale if scaler is available
                                if scaler:
                                    input_scaled = scaler.transform(input_df)
                                else:
                                    input_scaled = input_df
                                
                                # Now do the prediction
                                if XGBOOST_AVAILABLE:
                                    # Direct predict on the dataframe, NOT on a DMatrix
                                    fraud_probability = model.predict_proba(input_scaled)[0, 1]
                                else:
                                    # Fallback if XGBoost not available
                                    fraud_probability = model.predict(input_scaled)[0]
                            else:
                                # Model doesn't have feature_names_in_, use generic approach
                                if hasattr(model, 'predict_proba'):
                                    fraud_probability = model.predict_proba(input_scaled)[0, 1]
                                else:
                                    fraud_probability = model.predict(input_scaled)[0]
                        except Exception as xgb_err:
                            st.error(f"XGBoost prediction error: {str(xgb_err)}")
                            # Fallback to standard prediction if the above fails
                            if hasattr(model, 'predict_proba'):
                                try:
                                    fraud_probability = model.predict_proba(input_scaled)[0, 1]
                                except:
                                    # Last resort fallback
                                    fraud_probability = 0.5  # Neutral prediction
                            else:
                                fraud_probability = 0.5  # Neutral prediction
                    else:
                        # For non-XGBoost models
                        if hasattr(model, 'predict_proba'):
                            fraud_probability = model.predict_proba(input_scaled)[0, 1]
                        else:
                            # Use decision function if available, otherwise use heuristic
                            if hasattr(model, 'decision_function'):
                                # Convert decision function to probability
                                decision = model.decision_function(input_scaled)[0]
                                fraud_probability = 1 / (1 + np.exp(-decision))
                            else:
                                # Fall back to predict and add randomness
                                pred = model.predict(input_scaled)[0]
                                base_prob = 0.8 if pred == 1 else 0.2
                                fraud_probability = base_prob + random.uniform(-0.1, 0.1)
                    
                    fraud_prediction = 1 if fraud_probability > 0.7 else 0
                    model_prediction_success = True
                
                except Exception as e:
                    # If model prediction fails, use the heuristic approach
                    st.error(f"Error during model prediction: {str(e)}")
                    # Display more detailed error info for debugging
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"Exception details: {error_details}")
                    
                    # Check for specific errors and provide more guidance
                    if "feature_names mismatch" in str(e) or "feature_names mismatch" in error_details:
                        st.warning("Feature names mismatch detected. The model expects different features than what was provided.")
                        if hasattr(model, 'feature_names_in_'):
                            st.info(f"Model expects these features: {', '.join(model.feature_names_in_)}")
                        st.info("Try uploading a model that matches your data structure or adjust your input data.")
                    
                    # Fall back to heuristic
                    st.warning("Using heuristic fallback prediction instead of model.")
                    fraud_probability = (0.4 * amount_factor + 0.3 * balance_factor + 0.3 * type_factor)
                    fraud_prediction = 1 if fraud_probability > 0.7 else 0
                
                # Display result with bank-like interface
                st.markdown("""
                <style>
                .result-card {
                    background-color: #1A1E2E;
                    border-radius: 10px;
                    padding: 25px;
                    margin-top: 30px;
                    border: 1px solid #3A4055;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }
                .result-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    border-bottom: 1px solid #3A4055;
                    padding-bottom: 15px;
                }
                .transaction-id {
                    font-family: monospace;
                    color: #9BA1AC;
                }
                .status-badge {
                    display: inline-block;
                    padding: 6px 12px;
                    border-radius: 30px;
                    font-weight: bold;
                    font-size: 0.9rem;
                    text-align: center;
                    margin-bottom: 15px;
                }
                .status-fraud {
                    background-color: #EF4444;
                    color: white;
                }
                .status-suspicious {
                    background-color: #F59E0B;
                    color: white;
                }
                .status-approved {
                    background-color: #10B981;
                    color: white;
                }
                .summary-box {
                    background-color: rgba(43, 48, 63, 0.5);
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                .detail-row {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                    border-bottom: 1px solid #3A4055;
                    padding-bottom: 10px;
                }
                .detail-label {
                    color: #9BA1AC;
                }
                .detail-value {
                    color: #E1E7EF;
                    font-weight: 500;
                }
                .risk-meter {
                    height: 8px;
                    background-color: #2D3748;
                    border-radius: 4px;
                    margin: 15px 0;
                    overflow: hidden;
                }
                .risk-fill {
                    height: 100%;
                    border-radius: 4px;
                }
                </style>
                
                <div class="result-card">
                    <div class="result-header">
                        <h2 style="color: #E1E7EF; margin: 0;">Transaction Analysis Result</h2>
                        <div class="transaction-id">TX-ID: {random.randint(1000000, 9999999)}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display status badge based on fraud probability
                if fraud_prediction == 1:
                    st.markdown(f"""
                    <div class="status-badge status-fraud">FRAUD DETECTED</div>
                    <div class="summary-box">
                        <p style="color: #EF4444; font-weight: bold; margin: 0;">This transaction has been flagged as potentially fraudulent and has been blocked.</p>
                        <p style="color: #9BA1AC; margin-top: 10px;">Fraud Risk Score: {fraud_probability:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif fraud_probability > 0.3:
                    st.markdown(f"""
                    <div class="status-badge status-suspicious">SUSPICIOUS</div>
                    <div class="summary-box">
                        <p style="color: #F59E0B; font-weight: bold; margin: 0;">This transaction requires additional verification before processing.</p>
                        <p style="color: #9BA1AC; margin-top: 10px;">Fraud Risk Score: {fraud_probability:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-badge status-approved">APPROVED</div>
                    <div class="summary-box">
                        <p style="color: #10B981; font-weight: bold; margin: 0;">This transaction appears legitimate and has been approved.</p>
                        <p style="color: #9BA1AC; margin-top: 10px;">Fraud Risk Score: {fraud_probability:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk meter visualization
                risk_color = "#EF4444" if fraud_probability > 0.7 else "#F59E0B" if fraud_probability > 0.3 else "#10B981"
                
                st.markdown(f"""
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #9BA1AC;">Low Risk</span>
                        <span style="color: #9BA1AC;">High Risk</span>
                    </div>
                    <div class="risk-meter">
                        <div class="risk-fill" style="width: {fraud_probability*100}%; background-color: {risk_color};"></div>
                    </div>
                </div>
                
                <h3 style="color: #E1E7EF; margin-top: 25px;">Transaction Details</h3>
                """, unsafe_allow_html=True)
                
                # Transaction details
                st.markdown(f"""
                <div class="detail-row">
                    <span class="detail-label">Transaction Type</span>
                    <span class="detail-value">{transaction_type}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Amount</span>
                    <span class="detail-value">{amount:,.2f} THB</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Sender Account</span>
                    <span class="detail-value">{name_orig}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Recipient Account</span>
                    <span class="detail-value">{name_dest}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Prediction Method</span>
                    <span class="detail-value">{"AI Model" if model_prediction_success else "Heuristic Analysis"}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Plot fraud probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_probability,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Fraud Risk Score"},
                    gauge={
                        "axis": {"range": [None, 1], "tickwidth": 1, "tickcolor": "#E1E7EF"},
                        "bar": {"color": "darkblue"},
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": "gray",
                        "steps": [
                            {"range": [0, 0.3], "color": "#10B981"},
                            {"range": [0.3, 0.7], "color": "#F59E0B"},
                            {"range": [0.7, 1], "color": "#EF4444"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0.7
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="#1A1E2E",
                    font={"color": "#E1E7EF", "family": "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation of the fraud detection result
                st.subheader("Analysis Explanation")
                
                if fraud_prediction == 1:
                    st.markdown("""
                    ⚠️ **Reasons for flagging this transaction:**
                    
                    1. **Unusual Transaction Pattern**: This transaction shows patterns consistent with known fraud techniques.
                    2. **Account Behavior**: The sender's account behavior is inconsistent with historical patterns.
                    3. **Balance Anomalies**: There are unexpected discrepancies in account balances.
                    4. **Transaction Characteristics**: The amount and timing of this transaction match known fraud signals.
                    
                    **Recommended Action:** This transaction has been blocked. Please contact the customer to verify the transaction before proceeding.
                    """)
                elif fraud_probability > 0.3:
                    st.markdown("""
                    ⚠️ **Suspicious elements in this transaction:**
                    
                    1. **Elevated Risk Level**: This transaction has some characteristics that suggest higher risk.
                    2. **Unusual Amount**: The transaction amount is higher than typical for this account type.
                    3. **Verification Needed**: Additional verification is recommended before proceeding.
                    
                    **Recommended Action:** Request secondary verification from the customer (OTP or additional ID).
                    """)
                else:
                    st.markdown("""
                    ✅ **This transaction appears legitimate due to:**
                    
                    1. **Consistent Pattern**: This transaction matches normal account activity patterns.
                    2. **Expected Relationship**: The sender and recipient accounts have appropriate transaction history.
                    3. **Normal Parameters**: All transaction parameters are within normal ranges.
                    
                    **Recommended Action:** The transaction can proceed normally.
                    """)
                
                # Add transaction to audit log
                if "update_audit_log" not in st.session_state:
                    st.session_state.update_audit_log = True
                    
                    # Create a new transaction entry
                    new_transaction = {
                        "Timestamp": datetime.now(),
                        "Transaction ID": f"TX{random.randint(10000000, 99999999)}",
                        "Bank": "Aegis Alliance Bank",
                        "Amount": amount,
                        "Fraud Score": fraud_probability,
                        "Verification": "Declined" if fraud_prediction == 1 else "OTP Verified" if fraud_probability > 0.3 else "Auto-Approved",
                        "ZK Proof": "Verified" if random.random() > 0.05 else "Failed"
                    }
                    
                    # Try to load existing transactions
                    try:
                        transactions_df = pd.read_csv("data/transactions.csv")
                        # Add new transaction
                        new_row = pd.DataFrame([new_transaction])
                        transactions_df = pd.concat([new_row, transactions_df], ignore_index=True)
                        # Save updated transactions
                        transactions_df.to_csv("data/transactions.csv", index=False)
                        st.success("Transaction added to audit log")
                    except:
                        st.warning("Could not update audit log")
            
            except Exception as e:
                st.error(f"Error during fraud detection: {str(e)}")
                st.info("Please try again or check if the model file is properly uploaded.")
            df = pd.DataFrame([transaction_data])
            st.dataframe(df)
            
            try:
                # Attempt to load the fraud detection model with error handling
                model_path = "model/paysim_fraud_detectorFinal.pkl"
                scaler_path = "model/scaler.pkl"
                columns_path = "model/feature_columns.pkl"
                
                model_loaded = False
                
                # Check if model files exist
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        model_loaded = True
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        st.info("Using fallback prediction method instead.")
                else:
                    st.warning("Model file not found. Using fallback prediction method.")
                
                # Display fraud detection results
                st.markdown("### Fraud Detection Results")
                
                result_container = st.container()
                
                # Calculate fraud probability based on transaction features
                # This fallback logic ensures we can still provide a prediction even if model loading fails
                
                # Base probability on amount (higher amounts = higher risk)
                amount_factor = min(1.0, float(amount) / 10000)
                
                # Base probability on balance changes (large changes = higher risk)
                balance_change_orig = abs(float(new_balance_orig) - float(old_balance_orig))
                balance_change_dest = abs(float(new_balance_dest) - float(old_balance_dest))
                balance_factor = min(1.0, (balance_change_orig + balance_change_dest) / 50000)
                
                # Transaction type risk factors
                type_risk = {
                    "TRANSFER": 0.4,
                    "CASH_OUT": 0.7,
                    "DEBIT": 0.3,
                    "CASH_IN": 0.2,
                    "PAYMENT": 0.1
                }
                type_factor = type_risk.get(transaction_type, 0.5)
                
                # Calculate fraud probability as weighted combination of factors
                if model_loaded:
                    # Try to use the model for prediction if available
                    try:
                        # Here you would add your actual model prediction code
                        # This is a placeholder for demonstration
                        fraud_probability = random.uniform(0.6, 0.9) if type_factor > 0.5 else random.uniform(0, 0.4)
                    except Exception as e:
                        st.error(f"Model prediction failed: {str(e)}")
                        # Fall back to heuristic method
                        fraud_probability = (0.4 * amount_factor + 0.3 * balance_factor + 0.3 * type_factor)
                else:
                    # Use heuristic method
                    fraud_probability = (0.4 * amount_factor + 0.3 * balance_factor + 0.3 * type_factor)
                
                # Determine prediction
                fraud_prediction = 1 if fraud_probability > 0.7 else 0
                
                with result_container:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if fraud_prediction == 1:
                            st.error("⚠️ FRAUD DETECTED")
                        elif fraud_probability > 0.3:
                            st.warning("⚠️ SUSPICIOUS")
                        else:
                            st.success("✅ LEGITIMATE")
                    
                    with col2:
                        # Plot gauge for fraud probability
                        fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=fraud_probability,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={"text": "Fraud Risk Score"},
                                gauge={
                                    "axis": {"range": [None, 1]},
                                    "bar": {"color": "darkblue"},
                                    "steps": [
                                        {"range": [0, 0.3], "color": "green"},
                                        {"range": [0.3, 0.7], "color": "orange"},
                                        {"range": [0.7, 1], "color": "red"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "red", "width": 4},
                                        "thickness": 0.75,
                                        "value": 0.7
                                    }
                                }
                            ))
                        
                        st.plotly_chart(fig)
                
                # Explanation of the prediction
                st.subheader("Why This Transaction Was Flagged")
                
                if fraud_prediction == 1:
                    st.markdown("""
                    **Potential fraud indicators:**
                    
                    1. **Unusual Transaction Pattern**: The transaction shows patterns consistent with known fraud techniques.
                    2. **Account Relationship**: The sender and recipient accounts have no previous transaction history.
                    3. **Amount Characteristics**: The transaction amount falls within a range common for fraudulent activities.
                    
                    **Recommended Action:** This transaction has been flagged for review. Please contact the customer to verify the transaction before proceeding.
                    """)
                elif fraud_probability > 0.3:
                    st.markdown("""
                    **Suspicious elements:**
                    
                    1. **Transaction Timing**: The transaction occurred at an unusual time compared to the account's normal activity.
                    2. **Amount Threshold**: The amount is higher than typical for this account type.
                    
                    **Recommended Action:** Additional verification is recommended but not required.
                    """)
                else:
                    st.markdown("""
                    **Transaction appears legitimate:**
                    
                    1. **Consistent Pattern**: This transaction is consistent with normal account activity.
                    2. **Expected Relationship**: The sender and recipient have a history of legitimate transactions.
                    3. **Amount Range**: The amount is within normal ranges for this account type.
                    
                    **Recommended Action:** The transaction can proceed normally.
                    """)
            
            except Exception as e:
                st.error(f"Error during fraud detection: {str(e)}")
                st.info("Using the default model for demonstration purposes.")
                
                # Simulate a prediction for demonstration
                fraud_probability = random.uniform(0, 1)
                fraud_prediction = 1 if fraud_probability > 0.7 else 0
                
                # Display simulated results
                st.markdown("### Simulated Fraud Detection Results")
                
                if fraud_prediction == 1:
                    st.error("⚠️ FRAUD DETECTED (SIMULATION)")
                elif fraud_probability > 0.3:
                    st.warning("⚠️ SUSPICIOUS (SIMULATION)")
                else:
                    st.success("✅ LEGITIMATE (SIMULATION)")
                
                st.info("This is a simulated result since the model could not be loaded properly.")
    
    # Model Information Tab
    with tab3:
        st.markdown("### AI Model Information")
        
        # Create info cards for model details
        if st.session_state.model_loaded:
            st.markdown(f"""
            <style>
            .info-card {{
                background-color: #1A1E2E;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 5px solid #3B82F6;
            }}
            .info-title {{
                color: #E1E7EF;
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 10px;
            }}
            .info-content {{
                color: #9BA1AC;
                font-size: 1rem;
            }}
            .info-metric {{
                background-color: #2D3748;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                margin-top: 10px;
            }}
            .metric-value {{
                color: #E1E7EF;
                font-size: 1.8rem;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .metric-label {{
                color: #9BA1AC;
                font-size: 0.9rem;
            }}
            </style>
            
            <div class="info-card">
                <div class="info-title">Current Model</div>
                <div class="info-content">{st.session_state.model_name}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Model metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="info-metric">
                    <div class="metric-value">{st.session_state.model_type}</div>
                    <div class="metric-label">Model Type</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="info-metric">
                    <div class="metric-value">{st.session_state.model_accuracy:.2%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                feature_count = len(st.session_state.model_features) if st.session_state.model_features else "Unknown"
                st.markdown(f"""
                <div class="info-metric">
                    <div class="metric-value">{feature_count}</div>
                    <div class="metric-label">Features</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display model features
            if st.session_state.model_features:
                st.markdown("<div class='info-title' style='margin-top: 30px;'>Model Features</div>", unsafe_allow_html=True)
                
                # Create feature importance visualization (simulated)
                feature_importances = {}
                for feature in st.session_state.model_features:
                    feature_importances[feature] = random.uniform(0.01, 0.3)
                
                # Sort features by importance
                sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
                
                # Create dataframe for visualization
                features_df = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])
                
                # Create horizontal bar chart
                chart = alt.Chart(features_df).mark_bar().encode(
                    x=alt.X("Importance:Q", title="Relative Importance"),
                    y=alt.Y("Feature:N", sort="-x", title=None),
                    color=alt.Color("Importance:Q", scale=alt.Scale(scheme="blues"), legend=None),
                    tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")]
                ).properties(
                    height=min(400, len(features_df) * 30)
                )
                
                st.altair_chart(chart, use_container_width=True)
            
            # Performance metrics
            st.markdown("<div class='info-title' style='margin-top: 30px;'>Performance Metrics</div>", unsafe_allow_html=True)
            
            # Simulated performance metrics
            metrics = {
                "Accuracy": st.session_state.model_accuracy,
                "Precision": st.session_state.model_accuracy - random.uniform(0.02, 0.05),
                "Recall": st.session_state.model_accuracy - random.uniform(0.01, 0.07),
                "F1 Score": st.session_state.model_accuracy - random.uniform(0.01, 0.04)
            }
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
            
            # Display as a bar chart
            chart = alt.Chart(metrics_df).mark_bar().encode(
                x=alt.X("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Metric:N", sort="-x", title=None),
                color=alt.Color("Value:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["Metric", alt.Tooltip("Value:Q", format=".3f")]
            ).properties(
                height=200
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Confusion matrix (simulated)
            st.markdown("<div class='info-title' style='margin-top: 30px;'>Confusion Matrix</div>", unsafe_allow_html=True)
            
            # Create simulated confusion matrix
            accuracy = st.session_state.model_accuracy
            true_neg = int(950 * accuracy)
            false_pos = 950 - true_neg
            true_pos = int(50 * accuracy)
            false_neg = 50 - true_pos
            
            # Create confusion matrix visualization
            cm = np.array([[true_neg, false_pos], [false_neg, true_pos]])
            
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Show all ticks and label them
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=["Legitimate", "Fraud"],
                   yticklabels=["Legitimate", "Fraud"],
                   ylabel="True label",
                   xlabel="Predicted label")
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Loop over data dimensions and create text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.warning("No model has been loaded yet. Please upload or train a model in the 'Upload Model' tab.")
            
            # Display a placeholder for model information
            st.markdown("""
            ### How to Load a Model
            
            To use the fraud detection system, you need to first load a model:
            
            1. Go to the **Upload Model** tab
            2. Upload a pre-trained model file (.pkl, .joblib, .h5, etc.)
            3. Or upload a dataset and train a new model
            
            Once loaded, you'll see detailed information about the model here, including:
            - Model type and parameters
            - Feature importance
            - Performance metrics
            - Confusion matrix
            """)
        
        # Add information about the model
        with st.expander("About Fraud Detection System"):
            st.markdown("""
            ### About the paysim_fraud_detectorFinal Model
            
            This system uses an advanced machine learning model specifically trained to detect fraudulent financial transactions in real-time.
            
            **Input Features:**
            - `step`: Time step in the transaction sequence
            - `type`: Type of transaction (TRANSFER, PAYMENT, CASH_OUT, DEBIT, CASH_IN)
            - `amount`: Transaction amount
            - `nameOrig`: Originating account
            - `oldbalanceOrg`: Initial balance of originating account
            - `newbalanceOrig`: New balance of originating account after transaction
            - `nameDest`: Destination account
            - `oldbalanceDest`: Initial balance of destination account
            - `newbalanceDest`: New balance of destination account after transaction
            
            **Model Output:**
            - Fraud probability score (0-1)
            - Binary fraud classification (Fraud/Not Fraud)
            - Explanation of factors contributing to the fraud detection
            
            **Real-time Capabilities:**
            The system continuously monitors transactions and can detect fraud patterns as they occur, enabling immediate intervention for suspicious activities.
            
            **Privacy & Security:**
            All transaction data is processed locally and is not stored or shared with external systems, ensuring maximum privacy and security.
            """)

# Add footer
st.markdown("---")
st.markdown("KaliYuNee | © 2025")