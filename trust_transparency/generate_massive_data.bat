@echo off
echo ======================================================
echo Massive Transaction Dataset Generator (70 million records)
echo ======================================================
echo.
echo This script will generate a massive dataset of transactions
echo WARNING: This will create a very large file (~15-20GB)
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python first.
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import pandas; import numpy" > nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install pandas numpy
)

echo.
echo Starting transaction data generation...
echo This will take a significant amount of time.
echo.

REM Run the Python script
python generate_massive_transactions.py

echo.
echo Process completed.
echo.
pause
