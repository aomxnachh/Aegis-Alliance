#!/bin/bash

echo "Aegis Alliance - Trust & Transparency Layer Setup"
echo "================================================="

echo "Installing required packages..."
pip install -r requirements.txt

echo
echo "Generating sample data..."
python generate_data.py

echo
echo "Setup complete!"
echo
echo "To start the dashboard, run:"
echo "streamlit run dashboard.py"
echo

read -p "Press Enter to launch the dashboard..." 

echo "Starting Streamlit dashboard..."
streamlit run dashboard.py
