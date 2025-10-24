#!/usr/bin/env python3
"""
Daily Performance Tracking Script for Meta Ads

This script fetches daily performance data from Meta Ads API and updates
the daily_performance_tracking.csv file with comprehensive metrics.

Usage:
    python scripts/update_daily_performance.py [--date YYYY-MM-DD]
    
If no date is provided, it will fetch data for yesterday.
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import pandas as pd

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from integrations.meta_client import MetaClient, AccountAuth, ClientConfig
from integrations.slack import notify
from infrastructure.utils import getenv_f, getenv_i, getenv_b


def load_settings() -> Dict[str, Any]:
    """Load settings from environment variables."""
    return {
        'fb_app_id': os.getenv('FB_APP_ID'),
        'fb_app_secret': os.getenv('FB_APP_SECRET'),
        'fb_access_token': os.getenv('FB_ACCESS_TOKEN'),
        'fb_ad_account_id': os.getenv('FB_AD_ACCOUNT_ID'),
        'fb_pixel_id': os.getenv('FB_PIXEL_ID'),
        'fb_page_id': os.getenv('FB_PAGE_ID'),
        'ig_actor_id': os.getenv('IG_ACTOR_ID'),
        'store_url': os.getenv('STORE_URL'),
        'breakeven_cpa': getenv_f('BREAKEVEN_CPA', 50.0),
        'cogs_per_purchase': getenv_f('COGS_PER_PURCHASE', 20.0),
        'usd_eur_rate': getenv_f('USD_EUR_RATE', 0.85),
    }


def get_meta_client(settings: Dict[str, Any]) -> Optional[MetaClient]:
    """Initialize Meta API client."""
    try:
        auth = AccountAuth(
            app_id=settings['fb_app_id'],
            app_secret=settings['fb_app_secret'],
            access_token=settings['fb_access_token'],
            ad_account_id=settings['fb_ad_account_id'],
            pixel_id=settings['fb_pixel_id'],
            page_id=settings['fb_page_id'],
            instagram_actor_id=settings['ig_actor_id']
        )
        
        config = ClientConfig(
            api_version="v23.0",
            timeout=30,
            retries=3
        )
        
        return MetaClient(auth, config)
    except Exception as e:
        print(f"❌ Failed to initialize Meta client: {e}")
        return None


def fetch_daily_insights(meta_client: MetaClient, account_id: str, date: str) -> Dict[str, Any]:
    """Fetch daily insights from Meta API for a specific date."""
    try:
        # Format date for Meta API (YYYY-MM-DD)
        date_start = date
        date_end = date
        
        # Fetch account-level insights
        insights = meta_client.get_ad_insights(
            account_id=account_id,
            date_start=date_start,
            date_end=date_end,
            level="account"
        )
        
        if not insights or len(insights) == 0:
            print(f"⚠️ No insights data for {date}")
            return {}
        
        # Get the first (and should be only) insight record
        insight = insights[0]
        
        # Extract all relevant metrics
        metrics = {
            'spend_eur': float(insight.get('spend', 0)),
            'impressions': int(insight.get('impressions', 0)),
            'clicks': int(insight.get('clicks', 0)),
            'ctr_pct': float(insight.get('ctr', 0)) * 100 if insight.get('ctr') else 0,
            'cpc_eur': float(insight.get('cpc', 0)) if insight.get('cpc') else 0,
            'cpm_eur': float(insight.get('cpm', 0)) if insight.get('cpm') else 0,
            'purchases': int(insight.get('purchases', 0)),
            'atc': int(insight.get('add_to_cart', 0)),
            'ic': int(insight.get('initiate_checkout', 0)),
            'revenue_eur': float(insight.get('purchase_value', 0)) if insight.get('purchase_value') else 0,
            'frequency': float(insight.get('frequency', 0)) if insight.get('frequency') else 0,
            'reach': int(insight.get('reach', 0)),
            'three_sec_views': int(insight.get('video_3_sec_plays', 0)),
            'video_views': int(insight.get('video_plays', 0)),
            'watch_time_sec': float(insight.get('video_play_actions', 0)) if insight.get('video_play_actions') else 0,
            'dwell_time_sec': float(insight.get('link_clicks', 0)) if insight.get('link_clicks') else 0,  # Using link_clicks as proxy
        }
        
        # Calculate derived metrics
        if metrics['purchases'] > 0:
            metrics['roas'] = metrics['revenue_eur'] / metrics['spend_eur'] if metrics['spend_eur'] > 0 else 0
            metrics['cpa_eur'] = metrics['spend_eur'] / metrics['purchases']
            metrics['aov_eur'] = metrics['revenue_eur'] / metrics['purchases']
        else:
            metrics['roas'] = 0
            metrics['cpa_eur'] = 0
            metrics['aov_eur'] = 0
        
        # Calculate conversion rates
        if metrics['clicks'] > 0:
            metrics['conversion_rate_pct'] = (metrics['purchases'] / metrics['clicks']) * 100
            metrics['atc_rate_pct'] = (metrics['atc'] / metrics['clicks']) * 100
            metrics['ic_rate_pct'] = (metrics['ic'] / metrics['clicks']) * 100
        else:
            metrics['conversion_rate_pct'] = 0
            metrics['atc_rate_pct'] = 0
            metrics['ic_rate_pct'] = 0
        
        # Calculate funnel rates
        if metrics['atc'] > 0:
            metrics['atc_to_ic_rate_pct'] = (metrics['ic'] / metrics['atc']) * 100
        else:
            metrics['atc_to_ic_rate_pct'] = 0
            
        if metrics['ic'] > 0:
            metrics['ic_to_purchase_rate_pct'] = (metrics['purchases'] / metrics['ic']) * 100
        else:
            metrics['ic_to_purchase_rate_pct'] = 0
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error fetching insights for {date}: {e}")
        return {}


def get_ad_counts_by_stage(meta_client: MetaClient, account_id: str) -> Dict[str, int]:
    """Get count of ads by stage (testing, validation, scaling)."""
    try:
        # This is a simplified version - in reality you'd need to check ad names or labels
        # to determine which stage each ad is in
        ads = meta_client.get_ads(account_id=account_id)
        
        if not ads:
            return {'testing_ads': 0, 'validation_ads': 0, 'scaling_ads': 0, 'active_ads_count': 0}
        
        # For now, assume all active ads are in testing stage
        # In a real implementation, you'd check ad names, labels, or other indicators
        active_count = len([ad for ad in ads if ad.get('status') == 'ACTIVE'])
        
        return {
            'testing_ads': active_count,
            'validation_ads': 0,
            'scaling_ads': 0,
            'active_ads_count': active_count,
            'killed_ads': 0,
            'launched_ads': 0
        }
        
    except Exception as e:
        print(f"⚠️ Could not fetch ad counts: {e}")
        return {'testing_ads': 0, 'validation_ads': 0, 'scaling_ads': 0, 'active_ads_count': 0, 'killed_ads': 0, 'launched_ads': 0}


def update_csv_file(csv_path: str, date: str, metrics: Dict[str, Any], ad_counts: Dict[str, int], account_id: str) -> None:
    """Update the CSV file with new data for the specified date."""
    
    # Prepare the row data
    row_data = {
        'date': date,
        'account_id': account_id,
        'spend_eur': metrics.get('spend_eur', 0),
        'impressions': metrics.get('impressions', 0),
        'clicks': metrics.get('clicks', 0),
        'ctr_pct': round(metrics.get('ctr_pct', 0), 2),
        'cpc_eur': round(metrics.get('cpc_eur', 0), 2),
        'cpm_eur': round(metrics.get('cpm_eur', 0), 2),
        'purchases': metrics.get('purchases', 0),
        'atc': metrics.get('atc', 0),
        'ic': metrics.get('ic', 0),
        'revenue_eur': round(metrics.get('revenue_eur', 0), 2),
        'roas': round(metrics.get('roas', 0), 2),
        'cpa_eur': round(metrics.get('cpa_eur', 0), 2),
        'aov_eur': round(metrics.get('aov_eur', 0), 2),
        'conversion_rate_pct': round(metrics.get('conversion_rate_pct', 0), 2),
        'atc_rate_pct': round(metrics.get('atc_rate_pct', 0), 2),
        'ic_rate_pct': round(metrics.get('ic_rate_pct', 0), 2),
        'atc_to_ic_rate_pct': round(metrics.get('atc_to_ic_rate_pct', 0), 2),
        'ic_to_purchase_rate_pct': round(metrics.get('ic_to_purchase_rate_pct', 0), 2),
        'frequency': round(metrics.get('frequency', 0), 2),
        'reach': metrics.get('reach', 0),
        'three_sec_views': metrics.get('three_sec_views', 0),
        'video_views': metrics.get('video_views', 0),
        'watch_time_sec': round(metrics.get('watch_time_sec', 0), 2),
        'dwell_time_sec': round(metrics.get('dwell_time_sec', 0), 2),
        'quality_score': 0,  # Placeholder - would need ML system
        'stability_score': 0.0,  # Placeholder - would need ML system
        'momentum_score': 0.0,  # Placeholder - would need ML system
        'fatigue_index': 0.0,  # Placeholder - would need ML system
        'ad_age_days': 0,  # Placeholder - would need to track ad creation dates
        'active_ads_count': ad_counts.get('active_ads_count', 0),
        'testing_ads': ad_counts.get('testing_ads', 0),
        'validation_ads': ad_counts.get('validation_ads', 0),
        'scaling_ads': ad_counts.get('scaling_ads', 0),
        'killed_ads': ad_counts.get('killed_ads', 0),
        'launched_ads': ad_counts.get('launched_ads', 0),
        'notes': f"Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    }
    
    # Read existing CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_path}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return
    
    # Check if row for this date already exists
    existing_row = df[df['date'] == date]
    
    if not existing_row.empty:
        # Update existing row
        for col, value in row_data.items():
            df.loc[df['date'] == date, col] = value
        print(f"✅ Updated existing row for {date}")
    else:
        # Add new row
        new_row = pd.DataFrame([row_data])
        df = pd.concat([df, new_row], ignore_index=True)
        print(f"✅ Added new row for {date}")
    
    # Sort by date
    df = df.sort_values('date')
    
    # Save updated CSV
    try:
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV file updated: {csv_path}")
    except Exception as e:
        print(f"❌ Error saving CSV file: {e}")


def main():
    """Main function to update daily performance tracking."""
    parser = argparse.ArgumentParser(description='Update daily performance tracking CSV')
    parser.add_argument('--date', type=str, help='Date to fetch data for (YYYY-MM-DD). Defaults to yesterday.')
    parser.add_argument('--csv-path', type=str, default='data/daily_performance_tracking.csv', 
                       help='Path to the CSV file to update')
    
    args = parser.parse_args()
    
    # Determine the date to fetch data for
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        # Default to yesterday
        yesterday = datetime.now() - timedelta(days=1)
        target_date = yesterday.strftime('%Y-%m-%d')
    
    print(f"📊 Fetching performance data for {target_date}")
    
    # Load settings
    settings = load_settings()
    
    # Validate required settings
    required_vars = ['fb_app_id', 'fb_app_secret', 'fb_access_token', 'fb_ad_account_id']
    missing_vars = [var for var in required_vars if not settings.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Initialize Meta client
    meta_client = get_meta_client(settings)
    if not meta_client:
        sys.exit(1)
    
    # Fetch daily insights
    print(f"🔍 Fetching insights for account {settings['fb_ad_account_id']}")
    metrics = fetch_daily_insights(meta_client, settings['fb_ad_account_id'], target_date)
    
    if not metrics:
        print(f"⚠️ No metrics found for {target_date}")
        sys.exit(1)
    
    # Get ad counts by stage
    print("📈 Fetching ad counts by stage")
    ad_counts = get_ad_counts_by_stage(meta_client, settings['fb_ad_account_id'])
    
    # Update CSV file
    csv_path = os.path.join(os.path.dirname(__file__), '..', args.csv_path)
    print(f"💾 Updating CSV file: {csv_path}")
    update_csv_file(csv_path, target_date, metrics, ad_counts, settings['fb_ad_account_id'])
    
    # Print summary
    print(f"\n📊 Performance Summary for {target_date}:")
    print(f"   💰 Spend: €{metrics.get('spend_eur', 0):.2f}")
    print(f"   👀 Impressions: {metrics.get('impressions', 0):,}")
    print(f"   🖱️ Clicks: {metrics.get('clicks', 0):,}")
    print(f"   📈 CTR: {metrics.get('ctr_pct', 0):.2f}%")
    print(f"   🛒 Purchases: {metrics.get('purchases', 0)}")
    print(f"   💵 Revenue: €{metrics.get('revenue_eur', 0):.2f}")
    print(f"   📊 ROAS: {metrics.get('roas', 0):.2f}")
    print(f"   🎯 CPA: €{metrics.get('cpa_eur', 0):.2f}")
    print(f"   📱 Active Ads: {ad_counts.get('active_ads_count', 0)}")
    
    print(f"\n✅ Daily performance tracking updated successfully!")


if __name__ == "__main__":
    main()
