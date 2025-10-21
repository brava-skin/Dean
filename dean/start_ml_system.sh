#!/bin/bash

# Dean ML System Startup Script
# This script starts the ML-enhanced Dean system in production mode

echo "🚀 Starting Dean ML-Enhanced Meta Ads Automation System"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    echo "❌ Error: Please run this script from the Dean project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Make sure to set up your environment variables"
fi

# Install/update dependencies
echo "📦 Installing/updating dependencies..."
pip3 install -r requirements.txt --quiet

# Test ML system
echo "🧠 Testing ML system..."
python3 -c "
import xgboost as xgb
import sklearn
import pandas as pd
import numpy as np
print('✅ ML packages ready')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ ML system ready"
else
    echo "⚠️  ML system not fully ready, will use fallback"
fi

# Start the system
echo "🎯 Starting Dean ML system..."
echo "   Mode: ML-Enhanced (default)"
echo "   Fallback: Legacy system if ML unavailable"
echo ""

# Run the system with ML mode enabled by default
python3 src/main.py "$@"
