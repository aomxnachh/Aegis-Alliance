@echo off
echo Starting Real-time Fraud Detection Dashboard...
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import streamlit" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install requirements. Please check your internet connection.
        pause
        exit /b 1
    )
)

REM Launch the dashboard
echo Launching dashboard...
start "" http://localhost:8501
streamlit run dashboard.py

pause
