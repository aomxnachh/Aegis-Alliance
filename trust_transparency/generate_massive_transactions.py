import pandas as pd
import numpy as np
import random
import uuid
import datetime
import os
import time
import sys

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        print()

def generate_transaction_chunk(chunk_size, start_idx):
    """Generate a chunk of transaction data"""
    # Define lists of possible values
    banks = ["Bank A", "Bank B", "Bank C"]
    verification_statuses = ["Verified", "Pending", "Failed"]
    
    # Generate random data
    transactions = {
        "Timestamp": [
            (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 90), 
                                                         hours=random.randint(0, 23), 
                                                         minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(chunk_size)
        ],
        "Transaction ID": [str(uuid.uuid4()) for _ in range(chunk_size)],
        "Bank": [random.choice(banks) for _ in range(chunk_size)],
        "type": [random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]) for _ in range(chunk_size)],
        "Amount": [round(random.uniform(10.0, 100000.0), 2) for _ in range(chunk_size)],
        "nameOrig": [f"C{random.randint(1000000, 9999999)}" for _ in range(chunk_size)],
        "oldbalanceOrg": [round(random.uniform(0, 1000000.0), 2) for _ in range(chunk_size)],
        "newbalanceOrig": [round(random.uniform(0, 1000000.0), 2) for _ in range(chunk_size)],
        "nameDest": [f"C{random.randint(1000000, 9999999)}" for _ in range(chunk_size)],
        "oldbalanceDest": [round(random.uniform(0, 1000000.0), 2) for _ in range(chunk_size)],
        "newbalanceDest": [round(random.uniform(0, 1000000.0), 2) for _ in range(chunk_size)],
        "isFraud": [random.choices([0, 1], weights=[0.997, 0.003])[0] for _ in range(chunk_size)],
        "isFlaggedFraud": [random.choices([0, 1], weights=[0.999, 0.001])[0] for _ in range(chunk_size)],
        "Fraud Score": [round(random.uniform(0.0, 1.0), 4) for _ in range(chunk_size)],
        "Verification": [random.choice(verification_statuses) for _ in range(chunk_size)],
        "ZK Proof": [f"zk_{uuid.uuid4().hex[:16]}" for _ in range(chunk_size)]
    }
    
    # Ensure fraud scores align with isFraud flag
    for i in range(chunk_size):
        if transactions["isFraud"][i] == 1:
            transactions["Fraud Score"][i] = round(random.uniform(0.7, 0.99), 4)
        else:
            transactions["Fraud Score"][i] = round(random.uniform(0.01, 0.3), 4)
    
    # Create DataFrame
    return pd.DataFrame(transactions)

def generate_and_save_transactions(total_transactions, output_file, chunk_size=100000):
    """Generate transactions in chunks and save to CSV file"""
    # Calculate number of chunks
    num_chunks = total_transactions // chunk_size
    remaining = total_transactions % chunk_size
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create file and write header
    header = True
    mode = 'w'  # First chunk overwrites any existing file
    
    print(f"Generating {total_transactions:,} transactions in {num_chunks} chunks of {chunk_size:,} plus {remaining:,} remaining...")
    start_time = time.time()
    
    # Generate and save chunks
    for i in range(num_chunks):
        chunk_start = time.time()
        df_chunk = generate_transaction_chunk(chunk_size, i * chunk_size)
        df_chunk.to_csv(output_file, mode=mode, index=False, header=header)
        mode = 'a'  # Append for subsequent chunks
        header = False  # Only write header once
        
        # Update progress bar
        elapsed = time.time() - start_time
        rate = (i+1) * chunk_size / elapsed
        eta = (num_chunks - i - 1) * (elapsed / (i+1))
        
        # Format progress message
        progress_msg = f"Chunk {i+1}/{num_chunks}"
        status = f"Rate: {rate:.0f} records/sec | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
        print_progress_bar(i+1, num_chunks, prefix=progress_msg, suffix=status, length=40)
    
    # Handle any remaining transactions
    if remaining > 0:
        df_chunk = generate_transaction_chunk(remaining, num_chunks * chunk_size)
        df_chunk.to_csv(output_file, mode='a', index=False, header=False)
    
    total_time = time.time() - start_time
    print(f"\nFinished generating {total_transactions:,} transactions in {total_time:.2f} seconds!")
    print(f"Average rate: {total_transactions/total_time:.0f} records/second")
    print(f"File saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # Define parameters
    TOTAL_TRANSACTIONS = 70_000_000  # 70 million transactions
    CHUNK_SIZE = 500_000  # Process 500,000 transactions at a time to avoid memory issues
    OUTPUT_FILE = "data/massive_transactions.csv"
    
    # Allow command-line overrides
    import argparse
    parser = argparse.ArgumentParser(description='Generate massive transaction dataset')
    parser.add_argument('--total', type=int, default=TOTAL_TRANSACTIONS, 
                        help=f'Total number of transactions to generate (default: {TOTAL_TRANSACTIONS:,})')
    parser.add_argument('--chunk', type=int, default=CHUNK_SIZE, 
                        help=f'Number of transactions per chunk (default: {CHUNK_SIZE:,})')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, 
                        help=f'Output file path (default: {OUTPUT_FILE})')
    
    args = parser.parse_args()
    
    print(f"╔═══════════════════════════════════════════════════════════════╗")
    print(f"║                MASSIVE TRANSACTION GENERATOR                   ║")
    print(f"╚═══════════════════════════════════════════════════════════════╝")
    print(f"• Target: {args.total:,} transactions")
    print(f"• Chunk size: {args.chunk:,} transactions per batch")
    print(f"• Output: {args.output}")
    print(f"• Estimated file size: ~{(args.total * 250) / (1024**3):.1f} GB (varies based on actual data)")
    print()
    
    # Confirm before starting
    if args.total > 1_000_000:
        confirm = input(f"This will generate a very large dataset. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    # Generate the data
    start_time_total = time.time()
    generate_and_save_transactions(args.total, args.output, args.chunk)
    
    # Report file size
    if os.path.exists(args.output):
        file_size_bytes = os.path.getsize(args.output)
        file_size_gb = file_size_bytes / (1024 ** 3)
        print(f"Generated file size: {file_size_gb:.2f} GB")
        print(f"Total time: {(time.time() - start_time_total):.2f} seconds")
