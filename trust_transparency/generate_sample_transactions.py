import pandas as pd
import numpy as np
import random
import uuid
import datetime
import os

def generate_sample_transactions(num_records=1000, output_file="data/sample_transactions.csv"):
    """Generate a small sample of transaction data for testing"""
    # Define lists of possible values
    banks = ["Bank A", "Bank B", "Bank C"]
    verification_statuses = ["Verified", "Pending", "Failed"]
    transaction_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    
    # Generate random data
    transactions = {
        "Timestamp": [
            (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 90), 
                                                         hours=random.randint(0, 23), 
                                                         minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(num_records)
        ],
        "Transaction ID": [str(uuid.uuid4()) for _ in range(num_records)],
        "Bank": [random.choice(banks) for _ in range(num_records)],
        "type": [random.choice(transaction_types) for _ in range(num_records)],
        "Amount": [round(random.uniform(10.0, 100000.0), 2) for _ in range(num_records)],
        "nameOrig": [f"C{random.randint(1000000, 9999999)}" for _ in range(num_records)],
        "oldbalanceOrg": [round(random.uniform(0, 1000000.0), 2) for _ in range(num_records)],
        "newbalanceOrig": [round(random.uniform(0, 1000000.0), 2) for _ in range(num_records)],
        "nameDest": [f"C{random.randint(1000000, 9999999)}" for _ in range(num_records)],
        "oldbalanceDest": [round(random.uniform(0, 1000000.0), 2) for _ in range(num_records)],
        "newbalanceDest": [round(random.uniform(0, 1000000.0), 2) for _ in range(num_records)],
        "isFraud": [random.choices([0, 1], weights=[0.997, 0.003])[0] for _ in range(num_records)],
        "isFlaggedFraud": [random.choices([0, 1], weights=[0.999, 0.001])[0] for _ in range(num_records)],
        "Fraud Score": [round(random.uniform(0.0, 1.0), 4) for _ in range(num_records)],
        "Verification": [random.choice(verification_statuses) for _ in range(num_records)],
        "ZK Proof": [f"zk_{uuid.uuid4().hex[:16]}" for _ in range(num_records)]
    }
    
    # Ensure fraud scores align with isFraud flag
    for i in range(num_records):
        if transactions["isFraud"][i] == 1:
            transactions["Fraud Score"][i] = round(random.uniform(0.7, 0.99), 4)
        else:
            transactions["Fraud Score"][i] = round(random.uniform(0.01, 0.3), 4)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {num_records} sample transactions")
    print(f"File saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # Generate a small sample for testing
    generate_sample_transactions(num_records=1000, output_file="data/sample_transactions.csv")
