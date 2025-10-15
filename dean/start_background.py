#!/usr/bin/env python3
"""
Start the background scheduler for automated monitoring.
This script runs the automation in background mode with:
- Hourly ticks for testing/validation/scaling
- 3-hour summaries of metrics and active ads  
- Daily morning summaries
- Critical event alerts
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def main():
    """Start the background scheduler."""
    print("🤖 Starting Dean Background Scheduler...")
    print("This will run automated monitoring with:")
    print("  • Hourly ticks for testing/validation/scaling")
    print("  • 3-hour summaries of metrics and active ads")
    print("  • Daily morning summaries")
    print("  • Critical event alerts")
    print("")
    print("Press Ctrl+C to stop")
    print("")
    
    # Import and run the main function with background flag
    from main import main as main_func
    
    # Set the background argument
    sys.argv = ["main.py", "--background"]
    
    try:
        main_func()
    except KeyboardInterrupt:
        print("\n🛑 Background scheduler stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
