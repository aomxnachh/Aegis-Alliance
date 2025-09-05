import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.metrics import roc_curve, auc
import datetime
import time
import random
import json
import os

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
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
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
    .css-1d391kg {
        background-color: var(--secondary-bg) !important;
    }
    
    .css-1v3fvcr {
        background-color: var(--secondary-bg) !important;
    }
    
    .css-qbe2hs {
        color: var(--text-color) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--accent-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #5A46AE;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
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
logo_col, text_col = st.sidebar.columns([1, 3])

# Display logo in a container with custom styling
with logo_col:
    st.image("logo/AegisAllianceLogo4.png", width=80)

# Display text in second column
with text_col:
    st.markdown("""
    <div>
        <h1 style="margin: 0; padding: 0; color: #E1E7EF; font-size: 1.5rem; font-weight: 700;">Aegis Alliance</h1>
        <p style="margin: 0; padding: 0; color: #9BA1AC; font-size: 0.9rem;">Trust & Transparency</p>
    </div>
    """, unsafe_allow_html=True)

# Add separator
st.sidebar.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #2A3140;'>", unsafe_allow_html=True)

# Dashboard sections with icons
st.sidebar.markdown("### Navigation")
sections = {
    "Overview": "",
    "Model Performance": "",
    "Privacy Metrics": "",
    "Audit Log": "",
    "Federation Status": ""
}

# Create radio buttons with custom styling
selected_section = st.sidebar.radio(
    "Navigation",
    list(sections.keys()),
    format_func=lambda x: f"{sections[x]} {x}",
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #2A3140;'>", unsafe_allow_html=True)

# Controls section
st.sidebar.markdown("### Controls")

# Epsilon slider with custom styling
st.sidebar.markdown("""
<div style="margin-bottom: 0.5rem;">
    <label style="font-size: 0.9rem; color: #9BA1AC; font-weight: 500;">Privacy Budget (ε)</label>
    <div class="badge badge-info" style="float: right; margin-top: -5px;">Configuration</div>
</div>
""", unsafe_allow_html=True)

epsilon = st.sidebar.slider("Privacy Budget", min_value=0.1, max_value=10.0, value=1.0, step=0.1, 
                          help="Lower ε means more privacy but potentially lower accuracy",
                          label_visibility="collapsed")

# Display current epsilon value with custom styling
st.sidebar.markdown(f"""
<div style="text-align: center; padding: 0.5rem; background-color: rgba(110, 86, 207, 0.1); border-radius: 8px; margin-bottom: 1rem;">
    <span style="font-size: 1.2rem; font-weight: 600; color: #6E56CF;">ε = {epsilon:.1f}</span>
</div>
""", unsafe_allow_html=True)

# Bank federation selector with custom styling
st.sidebar.markdown("""
<div style="margin-bottom: 0.5rem; margin-top: 1rem;">
    <label style="font-size: 0.9rem; color: #9BA1AC; font-weight: 500;">Select Bank</label>
</div>
""", unsafe_allow_html=True)

banks = ["Bank A", "Bank B", "Bank C", "All Banks"]
selected_bank = st.sidebar.selectbox("Select Bank", banks, label_visibility="collapsed")

# Time period selector with custom styling
st.sidebar.markdown("""
<div style="margin-bottom: 0.5rem; margin-top: 1rem;">
    <label style="font-size: 0.9rem; color: #9BA1AC; font-weight: 500;">Time Period</label>
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
import datetime
current_time = datetime.datetime.now().strftime("%b %d, %Y %H:%M")
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
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'TransactionID': 'Transaction ID',
            'FraudScore': 'Fraud Score',
            'ZKProof': 'ZK Proof'
        })
        
        # Convert timestamp strings to datetime objects
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Sort by timestamp descending
        df = df.sort_values('Timestamp', ascending=False)
        
        # Select relevant columns
        audit_df = df[['Timestamp', 'Transaction ID', 'Bank', 'Amount', 'Fraud Score', 'Verification', 'ZK Proof']]
        
        return audit_df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Fall back to generating sample data if file doesn't exist
        return generate_sample_audit_log(100)

@st.cache_data
def generate_sample_audit_log(n_entries=100):
    """Generate sample audit log entries as fallback"""
    np.random.seed(42)
    
    timestamps = [datetime.datetime.now() - datetime.timedelta(minutes=i*15) for i in range(n_entries)]
    timestamps.sort(reverse=True)
    
    transaction_ids = [f"TX{random.randint(10000, 99999)}" for _ in range(n_entries)]
    banks = ["Bank A", "Bank B", "Bank C"]
    bank_names = [random.choice(banks) for _ in range(n_entries)]
    
    # Transaction amounts with some outliers
    amounts = np.random.lognormal(mean=5.0, sigma=1.2, size=n_entries)
    
    # Fraud scores (mostly low, some high)
    fraud_scores = np.clip(np.random.beta(0.5, 5.0, size=n_entries), 0, 1)
    
    # Verification status based on fraud score
    verifications = []
    for score in fraud_scores:
        if score > 0.8:
            verifications.append("Declined")
        elif score > 0.5:
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
    
    # Create dataframe
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

# ======= MAIN CONTENT =======
if selected_section == "Overview":
    # Page header with title and description
    st.markdown("""
    <div class="dashboard-header">
        <h1>Aegis Alliance Dashboard</h1>
        <p>Trust & Transparency Layer: Real-time insights into system performance, privacy, and federation status</p>
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
    st.markdown("<h1 class='main-header'>Model Performance</h1>", unsafe_allow_html=True)
    
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
    st.markdown("<h1 class='main-header'>Privacy Metrics</h1>", unsafe_allow_html=True)
    
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
    st.markdown("<h1 class='main-header'>Transaction Audit Log</h1>", unsafe_allow_html=True)
    
    # Generate audit log data
    audit_log = load_audit_log()
    
    # Filter by selected bank
    if selected_bank != "All Banks":
        audit_log = audit_log[audit_log["Bank"] == selected_bank]
    
    # Filter by time period
    if selected_period == "Last 24 hours":
        cutoff = datetime.datetime.now() - datetime.timedelta(days=1)
        audit_log = audit_log[audit_log["Timestamp"] > cutoff]
    elif selected_period == "Last 7 days":
        cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
        audit_log = audit_log[audit_log["Timestamp"] > cutoff]
    elif selected_period == "Last 30 days":
        cutoff = datetime.datetime.now() - datetime.timedelta(days=30)
        audit_log = audit_log[audit_log["Timestamp"] > cutoff]
    
    # Format the dataframe for display
    display_log = audit_log.copy()
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
    
    # Display audit log
    st.dataframe(styled_log, use_container_width=True)
    
    # Summary metrics
    st.markdown("<h2 class='sub-header'>Audit Summary</h2>", unsafe_allow_html=True)
    
    # Calculate metrics
    total_transactions = len(audit_log)
    flagged_count = len(audit_log[audit_log["Verification"] != "Auto-Approved"])
    declined_count = len(audit_log[audit_log["Verification"] == "Declined"])
    zkp_failed = len(audit_log[audit_log["ZK Proof"] == "Failed"])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", total_transactions)
    
    with col2:
        st.metric("Flagged for Review", flagged_count, f"{flagged_count/total_transactions:.1%}")
    
    with col3:
        st.metric("Declined Transactions", declined_count, f"{declined_count/total_transactions:.1%}")
    
    with col4:
        st.metric("ZKP Failures", zkp_failed, f"{zkp_failed/total_transactions:.1%}")
    
    # Verification status distribution
    st.markdown("<h2 class='sub-header'>Verification Status Distribution</h2>", unsafe_allow_html=True)
    
    # Calculate verification distribution
    verification_counts = audit_log["Verification"].value_counts().reset_index()
    verification_counts.columns = ["Status", "Count"]
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#D1FAE5', '#FEF3C7', '#FECACA']
    ax.pie(verification_counts["Count"], labels=verification_counts["Status"], autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

elif selected_section == "Federation Status":
    st.markdown("<h1 class='main-header'>Federation Status</h1>", unsafe_allow_html=True)
    
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

# Add footer
st.markdown("---")
st.markdown("KaliYuNee | © 2025")