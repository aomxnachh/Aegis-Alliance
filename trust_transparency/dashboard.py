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
    page_title="KaliYuNee Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #FFEDD5;
        padding: 0.2rem;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ======= SIDEBAR =======
st.sidebar.image("./logo/AegisAllianceLogo4.png", width=250)
st.sidebar.title("KaliYuNee Aegis Alliance")
st.sidebar.markdown("### Trust & Transparency Dashboard")

# Dashboard sections
sections = ["Overview", "Model Performance", "Privacy Metrics", "Audit Log", "Federation Status"]
selected_section = st.sidebar.radio("Navigation", sections)

# Epsilon (privacy budget) slider for simulation
epsilon = st.sidebar.slider("Privacy Budget (ε)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, 
                           help="Lower ε means more privacy but potentially lower accuracy")

# Bank federation selector
banks = ["Bank A", "Bank B", "Bank C", "All Banks"]
selected_bank = st.sidebar.selectbox("Select Bank", banks)

# Time period selector
time_periods = ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
selected_period = st.sidebar.selectbox("Time Period", time_periods)

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
    st.markdown("<h1 class='main-header'>KaliYuNee: Aegis Alliance - Trust & Transparency Layer</h1>", unsafe_allow_html=True)
    
    # KPI metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>89.2%</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Fraud Detection Rate</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>ε = {:.1f}</p>".format(epsilon), unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Privacy Budget</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>99.7%</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>ZK Proof Verification Rate</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System architecture diagram
    st.markdown("<h2 class='sub-header'>System Architecture</h2>", unsafe_allow_html=True)
    
    # Placeholder for architecture diagram
    architecture = """
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color=lightblue];
        
        Bank1 [label="Bank A Data"];
        Bank2 [label="Bank B Data"];
        Bank3 [label="Bank C Data"];
        
        Oracle [label="Oracle Engine\n(XGBoost)"];
        Adaptive [label="Adaptive Intervention\n(Policy Engine)"];
        Federated [label="Zero-Knowledge Fabric\n(Federated Learning)"];
        Trust [label="Trust & Transparency\n(Audit & Verification)"];
        
        {Bank1, Bank2, Bank3} -> Federated;
        Federated -> Oracle;
        Oracle -> Adaptive;
        {Oracle, Adaptive, Federated} -> Trust;
    }
    """
    
    st.graphviz_chart(architecture)
    
    # Current status
    st.markdown("<h2 class='sub-header'>System Status</h2>", unsafe_allow_html=True)
    st.info(f"The Aegis Alliance is currently using a privacy budget of ε = {epsilon}. The system is fully operational with all 3 banks participating in the federation.")

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
    display_log["Amount"] = display_log["Amount"].map("${:.2f}".format)
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
