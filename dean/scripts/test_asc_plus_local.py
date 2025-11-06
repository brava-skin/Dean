#!/usr/bin/env python3
"""
Local test script to run ASC+ campaign and debug creative/ad creation.
Run this to see detailed logs of why creatives aren't being added to campaign.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    # Try loading from dean/.env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
    else:
        print(f"⚠️  No .env file found, using system environment variables")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# Set up environment
os.environ.setdefault("ML_MODE", "true")

def main():
    """Run ASC+ campaign locally with detailed logging."""
    print("=" * 80)
    print("🧪 LOCAL ASC+ CAMPAIGN TEST")
    print("=" * 80)
    print()
    
    # Check required environment variables
    required_vars = [
        "FB_APP_ID",
        "FB_APP_SECRET", 
        "FB_ACCESS_TOKEN",
        "FB_AD_ACCOUNT_ID",
        "FB_PAGE_ID",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "OPENAI_API_KEY",
        "FLUX_API_KEY",
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print()
        print("💡 Set these in your .env file or export them before running:")
        print("   export FB_APP_ID=...")
        print("   export FB_ACCESS_TOKEN=...")
        print("   etc.")
        return 1
    
    print("✅ All required environment variables are set")
    print()
    
    # Check campaign/adset IDs
    campaign_id = os.getenv("ASC_PLUS_CAMPAIGN_ID", "120233669753230160")
    adset_id = os.getenv("ASC_PLUS_ADSET_ID", "120233669753240160")
    
    print(f"📋 Campaign ID: {campaign_id}")
    print(f"📋 Ad Set ID: {adset_id}")
    print()
    
    try:
        # Import after path setup
        from stages.asc_plus import run_asc_plus_tick
        from integrations.meta_client import MetaClient
        from creative.image_generator import create_image_generator
        from ml.ml_intelligence import create_ml_system, MLConfig
        
        # Initialize components
        print("🔧 Initializing components...")
        
        # Meta client
        meta_client = MetaClient(
            app_id=os.getenv("FB_APP_ID"),
            app_secret=os.getenv("FB_APP_SECRET"),
            access_token=os.getenv("FB_ACCESS_TOKEN"),
            ad_account_id=os.getenv("FB_AD_ACCOUNT_ID"),
        )
        print("✅ Meta client initialized")
        
        # ML system
        ml_system = create_ml_system(
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        )
        print("✅ ML system initialized")
        
        # Image generator
        image_generator = create_image_generator(
            flux_api_key=os.getenv("FLUX_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            ml_system=ml_system,
        )
        print("✅ Image generator initialized")
        print()
        
        # Load settings
        print("📁 Loading configuration...")
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        rules_path = Path(__file__).parent.parent / "config" / "rules.yaml"
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return 1
        
        with open(config_path) as f:
            settings = yaml.safe_load(f)
        
        if rules_path.exists():
            with open(rules_path) as f:
                rules = yaml.safe_load(f)
            # Deep merge
            def deep_merge(base, override):
                for key, value in override.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value
            deep_merge(settings, rules)
        
        # Set campaign/adset IDs in settings
        if "ids" not in settings:
            settings["ids"] = {}
        settings["ids"]["asc_plus_campaign_id"] = campaign_id
        settings["ids"]["asc_plus_adset_id"] = adset_id
        
        print("✅ Configuration loaded")
        print()
        
        # Run ASC+ tick
        print("=" * 80)
        print("🚀 RUNNING ASC+ CAMPAIGN TICK")
        print("=" * 80)
        print()
        
        # Create a simple store-like object for compatibility
        class SimpleStore:
            def get(self, key, default=None):
                return default
            def set(self, key, value):
                pass
        
        store = SimpleStore()
        
        # Get rules separately
        rules = {}
        if rules_path.exists():
            with open(rules_path) as f:
                rules = yaml.safe_load(f)
        
        run_asc_plus_tick(
            client=meta_client,
            settings=settings,
            rules=rules,
            store=store,
            ml_system=ml_system,
        )
        
        print()
        print("=" * 80)
        print("✅ ASC+ CAMPAIGN TICK COMPLETED")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

