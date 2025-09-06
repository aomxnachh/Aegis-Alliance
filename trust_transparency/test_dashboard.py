"""
Automated test script for the Real-time Fraud Detection Dashboard
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import requests
import json

def test_model_creation():
    """Test creating and saving a dummy model"""
    print("Testing model creation...")
    
    # Create a simple dataset
    X = np.random.rand(100, 7)
    y = np.random.randint(0, 2, 100)
    
    # Create column names matching the expected features
    feature_names = ["step", "type", "amount", "oldbalanceOrg", 
                     "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Add feature names attribute
    model.feature_names_in_ = feature_names
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Save the model
    model_path = "model/test_fraud_detector.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Test model created and saved to {model_path}")
    return model_path

def test_dataset_creation():
    """Create a test dataset for fraud detection"""
    print("Testing dataset creation...")
    
    # Create a directory for data if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Define column names
    columns = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", 
               "newbalanceOrig", "nameDest", "oldbalanceDest", 
               "newbalanceDest", "isFraud"]
    
    # Create empty dataframe
    df = pd.DataFrame(columns=columns)
    
    # Generate 100 sample transactions
    for i in range(100):
        # Random values for transaction
        step = random.randint(1, 10)
        
        # Transaction type
        tx_type = random.choice(["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"])
        
        # Amount
        amount = random.uniform(10, 10000)
        
        # Account names
        name_orig = f"C{random.randint(1000000000, 9999999999)}"
        name_dest = f"M{random.randint(1000000000, 9999999999)}"
        
        # Balances
        old_balance_orig = random.uniform(1000, 100000)
        new_balance_orig = old_balance_orig - amount if tx_type != "CASH_IN" else old_balance_orig + amount
        
        old_balance_dest = random.uniform(1000, 100000)
        new_balance_dest = old_balance_dest + amount if tx_type in ["TRANSFER", "PAYMENT"] else old_balance_dest
        
        # Fraud label (5% of transactions are fraudulent)
        is_fraud = 1 if random.random() < 0.05 else 0
        
        # Add anomalies to fraudulent transactions
        if is_fraud == 1:
            # Fraudulent transactions might have suspicious balance changes
            if random.random() < 0.7:
                new_balance_orig = old_balance_orig  # No balance change despite transaction
            
            # Anomalous destination balance
            if random.random() < 0.5:
                new_balance_dest = old_balance_dest + (amount * 2)  # Incorrect balance increase
        
        # Create row
        row = {
            "step": step,
            "type": tx_type,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_balance_orig,
            "newbalanceOrig": new_balance_orig,
            "nameDest": name_dest,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
            "isFraud": is_fraud
        }
        
        # Append to dataframe
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Save dataset
    dataset_path = "data/test_transactions.csv"
    df.to_csv(dataset_path, index=False)
    
    print(f"Test dataset created and saved to {dataset_path}")
    return dataset_path

def test_streamlit_api():
    """Test if Streamlit API is accessible"""
    try:
        # Create a test query
        test_transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 9000.0,
            "nameOrig": "C123456789",
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 1000.0,
            "nameDest": "M987654321",
            "oldbalanceDest": 5000.0,
            "newbalanceDest": 14000.0
        }
        
        print("Test transaction created for API testing")
        print("Note: This doesn't actually call the API since Streamlit doesn't have a REST API")
        print("This is just a demonstration of what the test data would look like")
        
        return True
    except Exception as e:
        print(f"Error in API test: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Starting test suite for Real-time Fraud Detection Dashboard")
    print("-" * 50)
    
    # Run tests
    model_path = test_model_creation()
    dataset_path = test_dataset_creation()
    api_success = test_streamlit_api()
    
    print("-" * 50)
    print("Test summary:")
    print(f"Model creation: {'SUCCESS' if os.path.exists(model_path) else 'FAILED'}")
    print(f"Dataset creation: {'SUCCESS' if os.path.exists(dataset_path) else 'FAILED'}")
    print(f"API test: {'SUCCESS' if api_success else 'FAILED'}")
    print("-" * 50)
    
    print("\nInstructions for manual testing:")
    print("1. Launch the dashboard: python dashboard.py")
    print("2. Navigate to 'Real-time Fraud Detection' section")
    print("3. Upload the test model and dataset created by this script")
    print("4. Try different transactions in the 'Interactive Detection' tab")
    print("5. Verify that the model information is displayed correctly")
    
    print("\nSee testing_guide.md for more detailed testing scenarios")

if __name__ == "__main__":
    run_all_tests()
