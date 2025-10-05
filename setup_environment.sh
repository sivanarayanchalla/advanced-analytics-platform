#!/bin/bash

echo "Setting up Advanced Analytics Platform with MongoDB..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw,processed,exports}
mkdir -p models/{saved_models,pipelines}
mkdir -p logs/{app,training}

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env 2>/dev/null || echo "Please create .env file with your configuration"
fi

echo "Setup completed successfully!"
echo ""
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To run the application, execute: streamlit run app.py"
echo ""
echo "Don't forget to:"
echo "1. Update .env file with your MongoDB Atlas connection string"
echo "2. Add your API keys for OpenAI and HuggingFace"