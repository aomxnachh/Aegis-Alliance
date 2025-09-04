import pandas as pd
import numpy as np
import datetime
import random
import json
import os

# Ensure directories exist
os.makedirs('data', exist_ok=True)

def generate_transactions(n_samples=5000):
    """Generate synthetic transaction data for demonstration"""
    np.random.seed(42)
    
    # Transaction timestamps (recent few days)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    timestamps = [start_date + (end_date - start_date) * random.random() for _ in range(n_samples)]
    timestamps.sort()
    
    # Transaction IDs
    transaction_ids = [f"TX{random.randint(10000, 99999)}" for _ in range(n_samples)]
    
    # Banks
    banks = ["Bank A", "Bank B", "Bank C"]
    bank_weights = [0.4, 0.3, 0.3]  # Different market shares
    bank_names = random.choices(banks, weights=bank_weights, k=n_samples)
    
    # Transaction amounts (lognormal distribution)
    amounts = np.random.lognormal(mean=5.0, sigma=1.2, size=n_samples)
    
    # User IDs (some users have multiple transactions)
    n_users = int(n_samples * 0.7)  # 70% of transactions are unique users
    user_ids = [f"USER{random.randint(1000, 9999)}" for _ in range(n_users)]
    transaction_users = random.choices(user_ids, k=n_samples)
    
    # Merchant categories
    categories = ["Retail", "Food", "Travel", "Online Services", "Entertainment", "Other"]
    category_weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]
    transaction_categories = random.choices(categories, weights=category_weights, k=n_samples)
    
    # Location
    countries = ["TH", "US", "SG", "JP", "OTHER"]
    country_weights = [0.8, 0.05, 0.05, 0.05, 0.05]
    transaction_countries = random.choices(countries, weights=country_weights, k=n_samples)
    
    # Device type
    devices = ["Mobile", "Web", "ATM", "POS", "Other"]
    device_weights = [0.6, 0.2, 0.1, 0.05, 0.05]
    transaction_devices = random.choices(devices, weights=device_weights, k=n_samples)
    
    # Fraud (imbalanced - only small percentage are fraudulent)
    # Base fraud rate
    base_fraud_rate = 0.005  # 0.5% base fraud rate
    
    # Adjust fraud probability based on features
    fraud_probs = []
    for i in range(n_samples):
        prob = base_fraud_rate
        
        # Higher fraud rate for higher amounts
        if amounts[i] > np.percentile(amounts, 95):
            prob *= 5
        
        # Higher fraud rate for certain countries
        if transaction_countries[i] == "OTHER":
            prob *= 3
        
        # Higher fraud rate for certain categories
        if transaction_categories[i] == "Online Services":
            prob *= 2
        
        fraud_probs.append(min(prob, 1.0))
    
    # Generate fraud labels based on calculated probabilities
    is_fraud = [random.random() < p for p in fraud_probs]
    
    # Model scores (correlated with fraud, but not perfect)
    fraud_scores = []
    for i in range(n_samples):
        if is_fraud[i]:
            # For fraudulent transactions, scores are generally high but with some errors
            score = np.random.beta(8, 2)  # Mostly high scores
        else:
            # For legitimate transactions, scores are generally low but with some false positives
            score = np.random.beta(2, 8)  # Mostly low scores
        fraud_scores.append(score)
    
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
    for _ in range(n_samples):
        r = random.random()
        if r > 0.02:  # 98% verification rate
            zk_proofs.append("Verified")
        else:
            zk_proofs.append("Failed")
    
    # Create dataframe
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "TransactionID": transaction_ids,
        "UserID": transaction_users,
        "Bank": bank_names,
        "Amount": amounts,
        "Category": transaction_categories,
        "Country": transaction_countries,
        "Device": transaction_devices,
        "IsFraud": is_fraud,
        "FraudScore": fraud_scores,
        "Verification": verifications,
        "ZKProof": zk_proofs
    })
    
    return df

def generate_model_metrics():
    """Generate privacy vs. performance metrics for different models"""
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    models = ["Bank A", "Bank B", "Bank C", "Federated"]
    
    metrics = []
    
    # Base metrics for each model at high epsilon (minimal privacy)
    base_metrics = {
        "Bank A": {"AUC": 0.82, "Precision": 0.75, "Recall": 0.71, "F1": 0.73, "Latency": 12},
        "Bank B": {"AUC": 0.79, "Precision": 0.72, "Recall": 0.68, "F1": 0.70, "Latency": 15},
        "Bank C": {"AUC": 0.84, "Precision": 0.78, "Recall": 0.73, "F1": 0.75, "Latency": 13},
        "Federated": {"AUC": 0.89, "Precision": 0.83, "Recall": 0.81, "F1": 0.82, "Latency": 18}
    }
    
    for eps in epsilon_values:
        for model in models:
            # Privacy impact factor (decreases with higher epsilon)
            privacy_impact = 1 - (0.3 / np.sqrt(eps)) if eps > 0 else 0
            privacy_impact = max(0.5, min(1.0, privacy_impact))
            
            metrics.append({
                "Model": model,
                "Epsilon": eps,
                "ROC-AUC": base_metrics[model]["AUC"] * privacy_impact,
                "Precision": base_metrics[model]["Precision"] * privacy_impact,
                "Recall": base_metrics[model]["Recall"] * privacy_impact,
                "F1-Score": base_metrics[model]["F1"] * privacy_impact,
                "Latency": base_metrics[model]["Latency"] * (1 + (0.5 / eps) if eps > 0 else 10)
            })
    
    return pd.DataFrame(metrics)

def generate_federation_progress():
    """Generate federated learning progress data"""
    rounds = 30
    models = ["Bank A", "Bank B", "Bank C", "Federated"]
    
    progress = []
    
    for r in range(1, rounds + 1):
        for model in models:
            # Base performance parameters
            if model == "Federated":
                base_auc = 0.7
                max_auc = 0.89
                learning_rate = 0.15
            elif model == "Bank A":
                base_auc = 0.65
                max_auc = 0.82
                learning_rate = 0.12
            elif model == "Bank B":
                base_auc = 0.63
                max_auc = 0.79
                learning_rate = 0.11
            else:  # Bank C
                base_auc = 0.67
                max_auc = 0.84
                learning_rate = 0.13
            
            # Learning curve (improvement over rounds)
            auc = base_auc + (max_auc - base_auc) * (1 - np.exp(-learning_rate * r))
            
            # Add some noise
            auc += np.random.uniform(-0.01, 0.01)
            
            progress.append({
                "Round": r,
                "Model": model,
                "ROC-AUC": auc,
                "Precision": auc * 0.9 + np.random.uniform(-0.02, 0.02),
                "Recall": auc * 0.85 + np.random.uniform(-0.02, 0.02),
                "F1-Score": auc * 0.87 + np.random.uniform(-0.02, 0.02)
            })
    
    return pd.DataFrame(progress)

# Generate all data
print("Generating transaction data...")
transactions = generate_transactions()
transactions.to_csv('data/transactions.csv', index=False)
print(f"Generated {len(transactions)} transactions")

print("Generating model metrics...")
model_metrics = generate_model_metrics()
model_metrics.to_csv('data/model_metrics.csv', index=False)
print(f"Generated metrics for {len(model_metrics)} model-epsilon combinations")

print("Generating federation progress...")
federation_progress = generate_federation_progress()
federation_progress.to_csv('data/federation_progress.csv', index=False)
print(f"Generated progress data for {federation_progress['Round'].nunique()} rounds")

print("Data generation complete. Files saved to the 'data' directory.")

# Also generate a summary JSON for quick loading
summary = {
    "transaction_count": len(transactions),
    "fraud_count": transactions['IsFraud'].sum(),
    "fraud_rate": transactions['IsFraud'].mean(),
    "verification_distribution": transactions['Verification'].value_counts().to_dict(),
    "zkp_verification_rate": (transactions['ZKProof'] == "Verified").mean(),
    "total_amount": transactions['Amount'].sum(),
    "fraudulent_amount": transactions[transactions['IsFraud']]['Amount'].sum(),
    "banks": transactions['Bank'].nunique(),
    "latest_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('data/summary.json', 'w') as f:
    json.dump(summary, f)

print("Summary JSON generated.")
