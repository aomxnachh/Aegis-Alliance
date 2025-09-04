# Aegis Alliance - Trust & Transparency Layer

## Overview
This dashboard provides transparency and trust metrics for the Aegis Alliance fraud detection system. It visualizes the trade-offs between privacy (epsilon values in Differential Privacy) and model performance (ROC-AUC), as well as providing an audit log of transactions.

## Features
- **Overview**: High-level system architecture and key metrics
- **Model Performance**: ROC curves and performance metrics at different privacy levels
- **Privacy Metrics**: Differential privacy and zero-knowledge proof statistics
- **Audit Log**: Transaction history with verification status
- **Federation Status**: Federated learning performance across participating banks

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run dashboard.py
```

## Usage
1. Use the sidebar to navigate between different sections
2. Adjust the privacy budget (ε) slider to see how different privacy levels affect performance
3. Select different banks or time periods to filter the audit log

## Privacy Budget (ε) Explained
- Lower ε values (0.1-1.0): Higher privacy, potentially lower accuracy
- Medium ε values (1.0-5.0): Balanced privacy and accuracy
- Higher ε values (5.0-10.0): Lower privacy, potentially higher accuracy

## Interactive Elements
- Adjust the privacy budget slider to see real-time changes in model performance
- Filter audit logs by bank and time period
- Explore federation performance metrics

## Visualization Types
- ROC curves showing model performance
- AUC vs Privacy Budget trade-off curves
- Verification status distribution
- Federated learning performance over training rounds
