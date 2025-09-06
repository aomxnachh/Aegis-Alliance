# Real-time Fraud Detection Dashboard Testing Guide

## Setup and Launch

1. Make sure you have all dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Launch the dashboard:
   ```
   python dashboard.py
   ```
   
   Or use the provided launch scripts:
   - On Windows: `launch.bat`
   - On Linux/Mac: `launch.sh`

## Testing Scenarios

### 1. Model Upload and Management

1. **Upload Pre-trained Model**:
   - Navigate to the "Real-time Fraud Detection" section
   - Select the "Upload Model" tab
   - Upload the model file from `model/paysim_fraud_detectorFinal.pkl`

2. **Upload Test Dataset**:
   - In the "Upload Model" tab, upload a CSV file with transaction data
   - You can use the existing `data/transactions.csv` for testing

3. **Train a New Model**:
   - After uploading a dataset, fill in the "New Model Name" field
   - Adjust test size and select an algorithm (e.g., "Random Forest")
   - Click "Train Model" and wait for the process to complete

### 2. Interactive Fraud Detection

1. **Test Legitimate Transaction**:
   - Navigate to the "Interactive Detection" tab
   - Enter the following values:
     - Step: 1
     - Transaction Type: "PAYMENT"
     - Amount: 500.00
     - Sender Account: "C1234567890"
     - Initial Balance: 100000.00
     - New Balance: 99500.00
     - Recipient Account: "M9876543210"
     - Recipient Initial Balance: 50000.00
     - Recipient New Balance: 50500.00
   - Submit the form and check the result (should show low fraud probability)

2. **Test Suspicious Transaction**:
   - Keep the same values but change:
     - Transaction Type: "CASH_OUT"
     - Amount: 5000.00
     - Initial Balance: 100000.00
     - New Balance: 95000.00
   - Submit the form (should show medium fraud probability)

3. **Test Fraudulent Transaction**:
   - Enter the following values:
     - Step: 1
     - Transaction Type: "CASH_OUT"
     - Amount: 50000.00
     - Sender Account: "C1234567890"
     - Initial Balance: 100000.00
     - New Balance: 100000.00 (unchanged balance is suspicious)
     - Recipient Account: "M9876543210"
     - Recipient Initial Balance: 1000.00
     - Recipient New Balance: 51000.00
   - Submit the form (should show high fraud probability)

4. **Test Balance Anomaly**:
   - Enter transaction where the balance changes don't match the amount:
     - Amount: 1000.00
     - Initial Balance: 100000.00
     - New Balance: 98000.00 (difference is 2000 instead of 1000)
   - This should trigger higher fraud probability

### 3. Model Information Verification

1. Navigate to the "Model Information" tab
2. Verify that the model details are displayed correctly:
   - Model name and type
   - Performance metrics
   - Feature importance visualization
   - Confusion matrix

3. Click on "About the Real-time Fraud Detection System" to view model details

## Troubleshooting

1. **If model loading fails**:
   - Verify the model path is correct
   - Check that the model file format is supported (.pkl, .joblib, .sav, .h5)
   - Try uploading a different model file

2. **If transaction validation fails**:
   - Ensure all required fields are filled
   - Check that numeric values are valid (no negative amounts)
   - Verify that balance changes are realistic

3. **If visualization doesn't appear**:
   - Check that you have the required libraries (matplotlib, altair, plotly)
   - Refresh the page and try again

## Expected Results

- The dashboard should successfully load and display all tabs
- Model upload should work and verify the model's validity
- Transaction analysis should provide a fraud score with explanations
- The visualization components should display properly
- The dashboard should handle errors gracefully and display appropriate messages
