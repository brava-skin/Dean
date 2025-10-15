#!/usr/bin/env python3
"""
Supabase Setup Helper for Dean Project

This script helps you set up your Supabase table with the correct schema.
Run this once to create the required table structure.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from main import setup_supabase_table, _get_supabase

def main():
    print("🔧 Dean Supabase Setup Helper")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if Supabase is configured
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not url or not key:
        print("❌ Supabase not configured!")
        print("\nPlease set these environment variables:")
        print("  SUPABASE_URL=your_supabase_url")
        print("  SUPABASE_SERVICE_ROLE_KEY=your_service_role_key")
        print("\nOr create a .env file with these values.")
        return False
    
    print(f"✅ Supabase URL: {url}")
    print(f"✅ Service Key: {'*' * 20}{key[-4:] if key else 'Not set'}")
    
    # Test connection
    sb = _get_supabase()
    if not sb:
        print("❌ Failed to connect to Supabase. Check your credentials.")
        return False
    
    print("✅ Connected to Supabase!")
    
    # Set up the table
    print("\n📋 Setting up table schema...")
    success = setup_supabase_table()
    
    if success:
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Go to your Supabase SQL editor")
        print("2. Run the provided SQL to create the table")
        print("3. Add some test data to your table")
        print("4. Run the Dean automation to test the connection")
    else:
        print("\n❌ Setup failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()
