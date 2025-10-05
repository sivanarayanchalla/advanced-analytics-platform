@echo off
echo Setting up Advanced Analytics Platform...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
mkdir data\raw data\processed data\exports 2>nul
mkdir models\saved_models models\pipelines 2>nul
mkdir logs\app logs\training 2>nul

echo.
echo Setup completed successfully!
echo.
echo To activate the virtual environment, run: venv\Scripts\activate.bat
echo To run the application, execute: streamlit run app.py
echo.
echo Please update the .env file with your configuration.
pause