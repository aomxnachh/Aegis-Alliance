import os
import sys
import subprocess
import webbrowser
import time

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import altair
        import sklearn
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def generate_data():
    """Generate sample data if it doesn't exist"""
    if not os.path.exists("data") or not os.path.exists("data/transactions.csv"):
        print("Generating sample data...")
        subprocess.run([sys.executable, "generate_data.py"])

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("Starting dashboard...")
    url = "http://localhost:8501"
    
    # Open browser after a short delay to allow Streamlit to start
    def open_browser():
        time.sleep(2)
        webbrowser.open(url)
    
    import threading
    threading.Thread(target=open_browser).start()
    
    # Start Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])

if __name__ == "__main__":
    print("Aegis Alliance - Trust & Transparency Layer")
    print("===========================================")
    
    if not check_requirements():
        print("Some required packages are missing.")
        choice = input("Do you want to install them now? (y/n): ")
        if choice.lower() == 'y':
            install_requirements()
        else:
            print("Cannot continue without required packages.")
            sys.exit(1)
    
    generate_data()
    launch_dashboard()
