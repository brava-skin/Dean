#!/bin/bash

# Daily Performance Tracking Script
# This script runs the daily performance tracking to update the CSV file

echo "📊 Starting Daily Performance Tracking..."

# Change to the dean directory
cd "$(dirname "$0")/.."

# Set the date (defaults to yesterday if not provided)
DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}

echo "📅 Fetching data for: $DATE"

# Run the Python script
python scripts/update_daily_performance.py --date "$DATE"

echo "✅ Daily performance tracking completed!"
