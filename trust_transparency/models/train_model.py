import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Create a sample model for fraud detection
def train_sample_model():
    """
    Creates a simple fraud detection model for demonstration purposes.
    This would be replaced by a properly trained model in production.
    """
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate random features
    step = np.random.randint(1, 100, n_samples)
    amount = np.random.lognormal(mean=5.0, sigma=2.0, size=n_samples)
    oldbalanceOrg = np.random.lognormal(mean=8.0, sigma=2.5, size=n_samples)
    newbalanceOrig = np.maximum(0, oldbalanceOrg - amount * np.random.uniform(0, 1.2, n_samples))
    oldbalanceDest = np.random.lognormal(mean=7.0, sigma=2.2, size=n_samples)
    newbalanceDest = oldbalanceDest + amount * np.random.uniform(0.8, 1.0, n_samples)
    
    # Transaction types
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    type_idx = np.random.randint(0, len(types), n_samples)
    transaction_type = [types[i] for i in type_idx]
    
    # Create rules for fraud based on patterns
    is_fraud = np.zeros(n_samples, dtype=int)
    
    # Rule 1: Large transfers where sender account is emptied
    for i in range(n_samples):
        # Large amount transfers that empty the account
        if (transaction_type[i] == 'TRANSFER' and 
            amount[i] > 1000000 and 
            (oldbalanceOrg[i] - newbalanceOrig[i]) / oldbalanceOrg[i] > 0.9):
            is_fraud[i] = 1
        
        # Unusual patterns of small amounts
        elif (transaction_type[i] in ['TRANSFER', 'CASH_OUT'] and 
              amount[i] < 10000 and 
              step[i] % 7 == 0 and 
              np.random.random() < 0.3):
            is_fraud[i] = 1
            
        # Random other frauds (very small percentage)
        elif np.random.random() < 0.005:
            is_fraud[i] = 1
    
    # Create a DataFrame
    data = pd.DataFrame({
        'step': step,
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFraud': is_fraud
    })
    
    # Add flagged fraud (not all fraud is flagged)
    data['isFlaggedFraud'] = 0
    data.loc[(data['isFraud'] == 1) & (np.random.random(n_samples) < 0.7), 'isFlaggedFraud'] = 1
    
    # Create dummy variables for transaction type
    data_with_dummies = pd.get_dummies(data, columns=['type'], drop_first=True)
    
    # Select features and target
    X = data_with_dummies.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = data_with_dummies['isFraud']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    os.makedirs('models', exist_ok=True)
    with open('models/fraud_detection_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Also save the column names for later use
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print("Model trained and saved to models/fraud_detection_model.pkl")
    return model, scaler, list(X.columns)

if __name__ == "__main__":
    train_sample_model()
