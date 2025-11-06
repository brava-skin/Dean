#!/usr/bin/env python3
"""
Simple debug script to run ASC+ campaign directly.
Avoids circular imports by running through main.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

# Change to dean directory
os.chdir(Path(__file__).parent.parent)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set campaign/adset IDs if not already set
if not os.getenv("ASC_PLUS_CAMPAIGN_ID"):
    os.environ["ASC_PLUS_CAMPAIGN_ID"] = "120233669753230160"
if not os.getenv("ASC_PLUS_ADSET_ID"):
    os.environ["ASC_PLUS_ADSET_ID"] = "120233669753240160"

# Run main.py which will handle everything
print("=" * 80)
print("🧪 RUNNING ASC+ CAMPAIGN DEBUG")
print("=" * 80)
print()
print(f"📋 Campaign ID: {os.getenv('ASC_PLUS_CAMPAIGN_ID')}")
print(f"📋 Ad Set ID: {os.getenv('ASC_PLUS_ADSET_ID')}")
print()
print("📝 Running main.py with detailed logging...")
print()

# Import and run main
if __name__ == "__main__":
    import main
    # main.py will handle the rest

