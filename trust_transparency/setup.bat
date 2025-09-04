@echo off
echo Aegis Alliance - Trust & Transparency Layer Setup
echo =================================================

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Generating sample data...
python generate_data.py

echo.
echo Setup complete!
echo.
echo To start the dashboard, run:
echo streamlit run dashboard.py
echo.
echo Press any key to launch the dashboard...
pause > nul

echo Starting Streamlit dashboard...
streamlit run dashboard.py
