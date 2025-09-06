"""
Generate test data with simulated fraud patterns for the Real-time Fraud Detection Dashboard
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_account_number(is_merchant=False):
    """Generate a random account number"""
    prefix = "M" if is_merchant else "C" 
    return f"{prefix}{random.randint(1000000000, 9999999999)}"

def generate_normal_transaction(step):
    """Generate a normal, non-fraudulent transaction"""
    # Select transaction type
    tx_type = random.choice(["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"])
    
    # Generate amount based on transaction type
    if tx_type == "PAYMENT":
        amount = random.uniform(10, 5000)
    elif tx_type == "TRANSFER":
        amount = random.uniform(50, 20000)
    elif tx_type == "CASH_OUT":
        amount = random.uniform(50, 10000)
    elif tx_type == "CASH_IN":
        amount = random.uniform(50, 8000)
    else:  # DEBIT
        amount = random.uniform(10, 2000)
    
    # Round amount to 2 decimal places
    amount = round(amount, 2)
    
    # Generate account names
    name_orig = generate_account_number(is_merchant=False)
    name_dest = generate_account_number(is_merchant=(tx_type in ["PAYMENT", "TRANSFER"]))
    
    # Generate balances
    old_balance_orig = random.uniform(amount * 1.5, amount * 10)
    new_balance_orig = old_balance_orig - amount if tx_type != "CASH_IN" else old_balance_orig + amount
    
    old_balance_dest = random.uniform(1000, 100000)
    new_balance_dest = old_balance_dest + amount if tx_type in ["TRANSFER", "PAYMENT"] else old_balance_dest
    
    # Round balances to 2 decimal places
    old_balance_orig = round(old_balance_orig, 2)
    new_balance_orig = round(new_balance_orig, 2)
    old_balance_dest = round(old_balance_dest, 2)
    new_balance_dest = round(new_balance_dest, 2)
    
    # Create transaction
    transaction = {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_balance_orig,
        "newbalanceOrig": new_balance_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "isFraud": 0,
        "isFlaggedFraud": 0
    }
    
    return transaction

def generate_fraudulent_transaction(step):
    """Generate a fraudulent transaction with suspicious patterns"""
    # Select fraud pattern
    fraud_pattern = random.choice([
        "balance_mismatch",
        "account_emptying",
        "large_transfer",
        "multiple_recipients",
        "unusual_merchant"
    ])
    
    if fraud_pattern == "balance_mismatch":
        # Transaction where balances don't add up correctly
        tx_type = "TRANSFER"
        amount = random.uniform(1000, 10000)
        amount = round(amount, 2)
        
        name_orig = generate_account_number(is_merchant=False)
        name_dest = generate_account_number(is_merchant=True)
        
        old_balance_orig = random.uniform(amount * 1.5, amount * 3)
        # Balance doesn't change despite transfer
        new_balance_orig = old_balance_orig
        
        old_balance_dest = random.uniform(1000, 5000)
        # Destination receives more than was sent
        new_balance_dest = old_balance_dest + (amount * 2)
        
        # Round balances
        old_balance_orig = round(old_balance_orig, 2)
        new_balance_orig = round(new_balance_orig, 2)
        old_balance_dest = round(old_balance_dest, 2)
        new_balance_dest = round(new_balance_dest, 2)
        
    elif fraud_pattern == "account_emptying":
        # Transaction that empties an account
        tx_type = "CASH_OUT"
        
        old_balance_orig = random.uniform(5000, 50000)
        # Amount is exactly equal to the balance (suspicious)
        amount = old_balance_orig
        
        name_orig = generate_account_number(is_merchant=False)
        name_dest = generate_account_number(is_merchant=False)
        
        # Account completely emptied
        new_balance_orig = 0
        
        old_balance_dest = random.uniform(1000, 5000)
        new_balance_dest = old_balance_dest
        
        # Round values
        old_balance_orig = round(old_balance_orig, 2)
        amount = round(amount, 2)
        new_balance_orig = round(new_balance_orig, 2)
        old_balance_dest = round(old_balance_dest, 2)
        new_balance_dest = round(new_balance_dest, 2)
        
    elif fraud_pattern == "large_transfer":
        # Unusually large transfer
        tx_type = "TRANSFER"
        
        # Very large amount
        amount = random.uniform(50000, 200000)
        
        name_orig = generate_account_number(is_merchant=False)
        name_dest = generate_account_number(is_merchant=True)
        
        # Balance just slightly higher than amount
        old_balance_orig = amount * 1.05
        new_balance_orig = old_balance_orig - amount
        
        old_balance_dest = random.uniform(10000, 50000)
        new_balance_dest = old_balance_dest + amount
        
        # Round values
        amount = round(amount, 2)
        old_balance_orig = round(old_balance_orig, 2)
        new_balance_orig = round(new_balance_orig, 2)
        old_balance_dest = round(old_balance_dest, 2)
        new_balance_dest = round(new_balance_dest, 2)
        
    elif fraud_pattern == "multiple_recipients":
        # Multiple small transfers (this is one of them)
        tx_type = "TRANSFER"
        
        # Small amount
        amount = random.uniform(500, 2000)
        
        name_orig = generate_account_number(is_merchant=False)
        name_dest = f"M{random.randint(1000000000, 1000000020)}"  # Suspicious merchant pattern
        
        old_balance_orig = random.uniform(50000, 100000)
        new_balance_orig = old_balance_orig - amount
        
        old_balance_dest = random.uniform(10000, 50000)
        new_balance_dest = old_balance_dest + amount
        
        # Round values
        amount = round(amount, 2)
        old_balance_orig = round(old_balance_orig, 2)
        new_balance_orig = round(new_balance_orig, 2)
        old_balance_dest = round(old_balance_dest, 2)
        new_balance_dest = round(new_balance_dest, 2)
        
    else:  # unusual_merchant
        # Payment to suspicious merchant
        tx_type = "PAYMENT"
        
        amount = random.uniform(5000, 15000)
        
        name_orig = generate_account_number(is_merchant=False)
        name_dest = "M9999999999"  # Suspicious merchant ID
        
        old_balance_orig = random.uniform(amount * 2, amount * 5)
        new_balance_orig = old_balance_orig - amount
        
        old_balance_dest = random.uniform(100000, 500000)  # Suspicious high balance
        new_balance_dest = old_balance_dest + amount
        
        # Round values
        amount = round(amount, 2)
        old_balance_orig = round(old_balance_orig, 2)
        new_balance_orig = round(new_balance_orig, 2)
        old_balance_dest = round(old_balance_dest, 2)
        new_balance_dest = round(new_balance_dest, 2)
    
    # Create transaction
    transaction = {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_balance_orig,
        "newbalanceOrig": new_balance_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "isFraud": 1,
        "isFlaggedFraud": 1 if amount > 50000 else 0
    }
    
    return transaction

def generate_test_dataset(num_transactions=1000, fraud_ratio=0.05):
    """Generate a test dataset with the specified number of transactions and fraud ratio"""
    print(f"Generating dataset with {num_transactions} transactions ({fraud_ratio*100:.1f}% fraudulent)...")
    
    transactions = []
    
    # Calculate number of fraudulent transactions
    num_fraud = int(num_transactions * fraud_ratio)
    num_normal = num_transactions - num_fraud
    
    print(f"- {num_normal} normal transactions")
    print(f"- {num_fraud} fraudulent transactions")
    
    # Generate transactions
    for step in range(1, num_normal + 1):
        transactions.append(generate_normal_transaction(step))
    
    for step in range(num_normal + 1, num_transactions + 1):
        transactions.append(generate_fraudulent_transaction(step))
    
    # Shuffle transactions
    random.shuffle(transactions)
    
    # Create dataframe
    df = pd.DataFrame(transactions)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"data/test_transactions_{timestamp}.csv"
    df.to_csv(filepath, index=False)
    
    print(f"Dataset saved to {filepath}")
    return filepath

def create_sample_transaction_batch():
    """Create a small batch of test transactions for manual testing"""
    print("Creating sample transactions for testing...")
    
    # Create a list of example transactions
    examples = []
    
    # 1. Normal PAYMENT
    examples.append({
        "description": "Normal PAYMENT transaction",
        "transaction": generate_normal_transaction(1)
    })
    
    # 2. Normal TRANSFER
    examples.append({
        "description": "Normal TRANSFER transaction",
        "transaction": {**generate_normal_transaction(2), "type": "TRANSFER"}
    })
    
    # 3. Normal CASH_OUT
    examples.append({
        "description": "Normal CASH_OUT transaction",
        "transaction": {**generate_normal_transaction(3), "type": "CASH_OUT"}
    })
    
    # 4. Fraudulent - Balance mismatch
    fraud_tx = generate_fraudulent_transaction(4)
    fraud_tx["type"] = "TRANSFER"
    fraud_tx["newbalanceOrig"] = fraud_tx["oldbalanceOrg"]  # Balance doesn't change
    examples.append({
        "description": "Fraudulent transaction - Balance mismatch",
        "transaction": fraud_tx
    })
    
    # 5. Fraudulent - Account emptying
    fraud_tx = generate_fraudulent_transaction(5)
    fraud_tx["type"] = "CASH_OUT"
    fraud_tx["amount"] = fraud_tx["oldbalanceOrg"]  # Amount equals balance
    fraud_tx["newbalanceOrig"] = 0  # Account emptied
    examples.append({
        "description": "Fraudulent transaction - Account emptying",
        "transaction": fraud_tx
    })
    
    # 6. Fraudulent - Very large transfer
    fraud_tx = generate_fraudulent_transaction(6)
    fraud_tx["type"] = "TRANSFER"
    fraud_tx["amount"] = 100000  # Very large amount
    examples.append({
        "description": "Fraudulent transaction - Very large transfer",
        "transaction": fraud_tx
    })
    
    # Print examples
    print("\nSample transactions for testing:")
    print("-" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['description']}")
        print("Transaction values to input:")
        tx = example['transaction']
        print(f"  Step: {tx['step']}")
        print(f"  Type: {tx['type']}")
        print(f"  Amount: {tx['amount']:.2f}")
        print(f"  Sender Account: {tx['nameOrig']}")
        print(f"  Initial Balance: {tx['oldbalanceOrg']:.2f}")
        print(f"  New Balance: {tx['newbalanceOrig']:.2f}")
        print(f"  Recipient Account: {tx['nameDest']}")
        print(f"  Recipient Initial Balance: {tx['oldbalanceDest']:.2f}")
        print(f"  Recipient New Balance: {tx['newbalanceDest']:.2f}")
        print(f"  Expected fraud label: {'Yes' if tx['isFraud'] == 1 else 'No'}")
        print("-" * 80)
    
    return examples

if __name__ == "__main__":
    print("Financial Fraud Detection - Test Data Generator")
    print("=" * 50)
    
    # Ask user what action to perform
    print("\nSelect an option:")
    print("1. Generate a full test dataset")
    print("2. Create sample transactions for manual testing")
    
    choice = input("\nEnter choice (1-2): ")
    
    if choice == "1":
        # Ask for parameters
        try:
            num_transactions = int(input("\nNumber of transactions to generate (default: 1000): ") or "1000")
            fraud_ratio = float(input("Fraud ratio (0-1, default: 0.05): ") or "0.05")
            
            # Generate dataset
            filepath = generate_test_dataset(num_transactions, fraud_ratio)
            
            print("\nDataset generation complete!")
            print(f"You can now use this dataset ({filepath}) to test the fraud detection dashboard.")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    
    elif choice == "2":
        # Create sample transactions
        examples = create_sample_transaction_batch()
        
        print("\nYou can use these sample transactions to test the fraud detection dashboard.")
        print("Copy the values into the transaction form in the 'Interactive Detection' tab.")
    
    else:
        print("Invalid choice. Please run the script again and select a valid option.")
