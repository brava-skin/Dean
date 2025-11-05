#!/usr/bin/env python3
"""
Check environment variables for Dean ASC+ Campaign
This script validates that all required environment variables are set.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
        print("   Or set environment variables directly in your shell.\n")

# Required environment variables
REQUIRED_ENVS = {
    "Meta Ads API": [
        "FB_APP_ID",
        "FB_APP_SECRET",
        "FB_ACCESS_TOKEN",
        "FB_AD_ACCOUNT_ID",
        "FB_PIXEL_ID",
        "FB_PAGE_ID",
        "IG_ACTOR_ID",
    ],
    "Supabase": [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",  # Preferred over SUPABASE_ANON_KEY
    ],
    "Creative Generation": [
        "FLUX_API_KEY",
        "OPENAI_API_KEY",
    ],
    "Optional": [
        "STORE_URL",
        "SLACK_WEBHOOK_URL",
        "CREATIVE_STORAGE_BUCKET",
        "ACCOUNT_TZ",
        "ACCOUNT_CURRENCY",
        "BREAKEVEN_CPA",
        "COGS_PER_PURCHASE",
        "USD_EUR_RATE",
    ],
}

# Alternative env vars (one or the other)
ALTERNATIVES = {
    "SUPABASE_SERVICE_ROLE_KEY": ["SUPABASE_ANON_KEY"],
    "ACCOUNT_TZ": ["ACCOUNT_TIMEZONE"],
}

def check_env() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Check which environment variables are set and missing."""
    missing = {}
    found = {}
    
    for category, vars_list in REQUIRED_ENVS.items():
        missing_vars = []
        found_vars = []
        
        for var in vars_list:
            value = os.getenv(var)
            
            # Check alternatives if this var is not set
            if not value and var in ALTERNATIVES:
                for alt_var in ALTERNATIVES[var]:
                    alt_value = os.getenv(alt_var)
                    if alt_value:
                        found_vars.append(f"{var} (using {alt_var})")
                        break
                else:
                    missing_vars.append(var)
            elif value:
                # Mask sensitive values
                if "KEY" in var or "SECRET" in var or "TOKEN" in var:
                    masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                    found_vars.append(f"{var}={masked}")
                else:
                    found_vars.append(f"{var}={value}")
            else:
                missing_vars.append(var)
        
        if missing_vars:
            missing[category] = missing_vars
        if found_vars:
            found[category] = found_vars
    
    return found, missing

def main():
    print("=" * 60)
    print("Dean ASC+ Campaign - Environment Variables Check")
    print("=" * 60)
    print()
    
    found, missing = check_env()
    
    # Print found variables
    if found:
        print("✅ FOUND VARIABLES:")
        print("-" * 60)
        for category, vars_list in found.items():
            if category != "Optional":
                print(f"\n{category}:")
                for var in vars_list:
                    print(f"  ✓ {var}")
        print()
    
    # Print missing variables
    if missing:
        print("❌ MISSING VARIABLES:")
        print("-" * 60)
        for category, vars_list in missing.items():
            if category != "Optional":
                print(f"\n{category}:")
                for var in vars_list:
                    print(f"  ✗ {var}")
        print()
        print("⚠️  Please add the missing variables to your .env file")
        print()
        return False
    
    # Print optional variables
    if "Optional" in found:
        print("ℹ️  OPTIONAL VARIABLES SET:")
        print("-" * 60)
        for var in found["Optional"]:
            print(f"  • {var}")
        print()
    
    print("✅ All required environment variables are set!")
    print()
    print("Next steps:")
    print("1. Verify Supabase schema is run (supabase_schema.sql)")
    print("2. Verify Supabase Storage bucket 'creatives' exists")
    print("3. Run Dean: python src/main.py")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

