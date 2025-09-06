# Real-time Fraud Detection Dashboard Testing Guide

## Overview

This document provides detailed instructions for testing the Real-time Fraud Detection Dashboard. The dashboard offers a comprehensive interface for uploading fraud detection models, analyzing transactions, and viewing model information.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Required packages: streamlit, pandas, numpy, matplotlib, seaborn, altair, scikit-learn, plotly

### Launch Options

**Option 1: Using the launcher script**
- Run `test_dashboard.bat` (Windows) or `test_dashboard.sh` (Linux/Mac)

**Option 2: Manual launch**
```powershell
# Install requirements first if needed
pip install -r requirements.txt

# Launch the dashboard
streamlit run dashboard.py
```

## Test Cases

### 1. Dashboard Navigation

**Test:** Navigate between different sections of the dashboard.
- Click on "Overview", "Model Performance", "Privacy Metrics", etc. in the sidebar
- Verify that the content changes accordingly
- Navigate to "Real-time Fraud Detection" to access the fraud detection interface

### 2. Model Upload and Management

#### 2.1 Upload Pre-trained Model

**Test:** Upload an existing model file.
1. Navigate to the "Real-time Fraud Detection" section
2. Select the "Upload Model" tab
3. Click "Browse files" in the model uploader
4. Select `model/paysim_fraud_detectorFinal.pkl`
5. Verify that a success message appears
6. Check that model info is updated (name, type)

**Expected Result:** Model is uploaded successfully and verification message is displayed.

#### 2.2 Upload Dataset

**Test:** Upload a dataset for training or testing.
1. In the "Upload Model" tab
2. Click "Browse files" in the dataset uploader
3. Select a CSV file (you can use data from `generate_test_data.py`)
4. Verify that a success message appears

**Expected Result:** Dataset is uploaded successfully and verification message is displayed.

#### 2.3 Train New Model

**Test:** Train a new model from an uploaded dataset.
1. After uploading a dataset, enter a name for the new model (e.g., "custom_fraud_model.pkl")
2. Adjust test size and select an algorithm (e.g., "Random Forest")
3. Click "Train Model"
4. Verify that the progress bar completes
5. Check that a success message appears with the model accuracy

**Expected Result:** Model training completes and the new model is available for use.

### 3. Transaction Testing

#### 3.1 Legitimate Transaction Test

**Test:** Process a transaction with low fraud risk.
1. Navigate to the "Interactive Detection" tab
2. Enter the following details:
   - Step: 1
   - Transaction Type: "PAYMENT"
   - Amount: 500.00
   - Sender Account: "C1234567890"
   - Initial Balance: 10000.00
   - New Balance: 9500.00
   - Recipient Account: "M9876543210"
   - Recipient Initial Balance: 5000.00
   - Recipient New Balance: 5500.00
3. Click "Verify and Process Transaction"
4. Observe the results

**Expected Result:** Transaction is classified as legitimate with a low fraud probability score (0-30%).

#### 3.2 Suspicious Transaction Test

**Test:** Process a transaction with medium fraud risk.
1. Navigate to the "Interactive Detection" tab
2. Enter the following details:
   - Step: 1
   - Transaction Type: "TRANSFER"
   - Amount: 8000.00
   - Sender Account: "C1234567890"
   - Initial Balance: 10000.00
   - New Balance: 2000.00
   - Recipient Account: "M9876543210"
   - Recipient Initial Balance: 1000.00
   - Recipient New Balance: 9000.00
3. Click "Verify and Process Transaction"
4. Observe the results

**Expected Result:** Transaction is classified as suspicious with a medium fraud probability score (30-70%).

#### 3.3 Fraudulent Transaction Test

**Test:** Process a transaction with high fraud risk.
1. Navigate to the "Interactive Detection" tab
2. Enter the following details:
   - Step: 1
   - Transaction Type: "CASH_OUT"
   - Amount: 9999.00
   - Sender Account: "C1234567890"
   - Initial Balance: 10000.00
   - New Balance: 10000.00  # No change in balance despite transaction
   - Recipient Account: "M9876543210"
   - Recipient Initial Balance: 1000.00
   - Recipient New Balance: 11000.00
3. Click "Verify and Process Transaction"
4. Observe the results

**Expected Result:** Transaction is classified as fraudulent with a high fraud probability score (70-100%).

#### 3.4 Balance Anomaly Test

**Test:** Process a transaction with inconsistent balance changes.
1. Navigate to the "Interactive Detection" tab
2. Enter the following details:
   - Step: 1
   - Transaction Type: "TRANSFER"
   - Amount: 1000.00
   - Sender Account: "C1234567890"
   - Initial Balance: 10000.00
   - New Balance: 8000.00  # Balance change is 2000 instead of 1000
   - Recipient Account: "M9876543210"
   - Recipient Initial Balance: 5000.00
   - Recipient New Balance: 7000.00  # Balance change is 2000 instead of 1000
3. Click "Verify and Process Transaction"
4. Observe the results

**Expected Result:** Transaction should be flagged as suspicious or fraudulent due to the balance inconsistencies.

### 4. Model Information Verification

**Test:** Check that model information is displayed correctly.
1. Navigate to the "Model Information" tab
2. Verify that model details are displayed (if a model is loaded)
3. Check that performance metrics and visualizations are shown
4. Expand the "About the Real-time Fraud Detection System" section
5. Verify that the model description is displayed correctly

**Expected Result:** Model information is displayed correctly with appropriate visualizations.

### 5. Error Handling Tests

#### 5.1 No Model Loaded

**Test:** Attempt to process a transaction without loading a model.
1. Launch a fresh instance of the dashboard
2. Navigate to "Real-time Fraud Detection" > "Interactive Detection"
3. Fill in transaction details
4. Click "Verify and Process Transaction"

**Expected Result:** System should use fallback heuristic methods and display a warning about no model being loaded.

#### 5.2 Invalid Input Values

**Test:** Enter invalid values in the transaction form.
1. Navigate to "Interactive Detection"
2. Enter invalid values (e.g., negative amount, text in numeric fields)
3. Try to submit the form

**Expected Result:** Form validation should prevent submission or handle errors gracefully.

#### 5.3 Incompatible Model Format

**Test:** Upload an incompatible file as a model.
1. Navigate to "Upload Model"
2. Upload a text file or non-model file
3. Observe the response

**Expected Result:** System should display an error message indicating the file is not a valid model.

## Using the Generated Test Data

For more comprehensive testing, you can use the `generate_test_data.py` script to create:

1. A full dataset of simulated transactions with known fraud patterns
2. Individual test cases with specific fraud characteristics

Run the script and follow the prompts to generate the data:

```powershell
python generate_test_data.py
```

## Automated Testing

The `test_dashboard.py` script provides basic automated tests for:
- Model creation and saving
- Test dataset generation
- API simulation

Run the automated tests with:

```powershell
python test_dashboard.py
```

## Troubleshooting

If you encounter issues during testing:

1. **Dashboard doesn't start:**
   - Check Python installation
   - Verify all requirements are installed
   - Check console for error messages

2. **Model upload fails:**
   - Verify the file format is supported (.pkl, .joblib, .h5, etc.)
   - Check file permissions
   - Ensure the model directory exists and is writable

3. **Visualizations don't appear:**
   - Refresh the page
   - Check browser console for errors
   - Verify that all required packages are installed
