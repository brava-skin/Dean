#!/usr/bin/env python3
"""
Meta Ads Performance Analysis Script

This script analyzes the daily_performance_tracking.csv file and provides
comprehensive insights and trends.

Usage:
    python scripts/analyze_performance.py [--csv-path path/to/file.csv] [--days N]
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_performance_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare the performance data."""
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return pd.DataFrame()


def calculate_trends(df: pd.DataFrame, days: int = 7) -> Dict[str, Dict]:
    """Calculate performance trends over the specified number of days."""
    if len(df) == 0:
        return {}
    
    # Get the last N days of data
    recent_data = df.tail(days)
    
    trends = {}
    
    # Key metrics to analyze
    metrics = [
        'spend_eur', 'impressions', 'clicks', 'ctr_pct', 'cpc_eur', 'cpm_eur',
        'purchases', 'atc', 'ic', 'revenue_eur', 'roas', 'cpa_eur', 'aov_eur',
        'conversion_rate_pct', 'atc_rate_pct', 'ic_rate_pct', 'active_ads_count'
    ]
    
    for metric in metrics:
        if metric in recent_data.columns:
            values = recent_data[metric].dropna()
            if len(values) > 1:
                # Calculate trend (slope of linear regression)
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                # Calculate percentage change
                if values.iloc[0] != 0:
                    pct_change = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0]) * 100
                else:
                    pct_change = 0
                
                trends[metric] = {
                    'current': values.iloc[-1],
                    'previous': values.iloc[0],
                    'trend': slope,
                    'pct_change': pct_change,
                    'avg': values.mean(),
                    'std': values.std()
                }
    
    return trends


def generate_performance_report(df: pd.DataFrame, days: int = 7) -> str:
    """Generate a comprehensive performance report."""
    if len(df) == 0:
        return "❌ No data available for analysis."
    
    report = []
    report.append("📊 META ADS PERFORMANCE ANALYSIS")
    report.append("=" * 50)
    report.append(f"📅 Analysis Period: Last {days} days")
    report.append(f"📈 Data Points: {len(df)} days")
    report.append("")
    
    # Overall performance summary
    recent_data = df.tail(days)
    total_spend = recent_data['spend_eur'].sum()
    total_revenue = recent_data['revenue_eur'].sum()
    total_purchases = recent_data['purchases'].sum()
    total_impressions = recent_data['impressions'].sum()
    total_clicks = recent_data['clicks'].sum()
    
    report.append("💰 FINANCIAL PERFORMANCE")
    report.append("-" * 30)
    report.append(f"Total Spend: €{total_spend:,.2f}")
    report.append(f"Total Revenue: €{total_revenue:,.2f}")
    report.append(f"Net Profit: €{total_revenue - total_spend:,.2f}")
    report.append(f"Overall ROAS: {total_revenue/total_spend:.2f}" if total_spend > 0 else "Overall ROAS: N/A")
    report.append(f"Average CPA: €{total_spend/total_purchases:.2f}" if total_purchases > 0 else "Average CPA: N/A")
    report.append("")
    
    # Traffic metrics
    report.append("📈 TRAFFIC METRICS")
    report.append("-" * 30)
    report.append(f"Total Impressions: {total_impressions:,}")
    report.append(f"Total Clicks: {total_clicks:,}")
    report.append(f"Average CTR: {recent_data['ctr_pct'].mean():.2f}%")
    report.append(f"Average CPC: €{recent_data['cpc_eur'].mean():.2f}")
    report.append(f"Average CPM: €{recent_data['cpm_eur'].mean():.2f}")
    report.append("")
    
    # Conversion metrics
    report.append("🛒 CONVERSION METRICS")
    report.append("-" * 30)
    report.append(f"Total Purchases: {total_purchases}")
    report.append(f"Total ATC: {recent_data['atc'].sum()}")
    report.append(f"Total IC: {recent_data['ic'].sum()}")
    report.append(f"Average Conversion Rate: {recent_data['conversion_rate_pct'].mean():.2f}%")
    report.append(f"Average ATC Rate: {recent_data['atc_rate_pct'].mean():.2f}%")
    report.append(f"Average IC Rate: {recent_data['ic_rate_pct'].mean():.2f}%")
    report.append("")
    
    # Funnel analysis
    if recent_data['atc'].sum() > 0 and recent_data['ic'].sum() > 0:
        atc_to_ic = (recent_data['ic'].sum() / recent_data['atc'].sum()) * 100
        ic_to_purchase = (recent_data['purchases'].sum() / recent_data['ic'].sum()) * 100
        report.append("🔄 FUNNEL ANALYSIS")
        report.append("-" * 30)
        report.append(f"ATC to IC Rate: {atc_to_ic:.2f}%")
        report.append(f"IC to Purchase Rate: {ic_to_purchase:.2f}%")
        report.append("")
    
    # Trends analysis
    trends = calculate_trends(df, days)
    if trends:
        report.append("📊 TREND ANALYSIS (Last 7 Days)")
        report.append("-" * 30)
        
        # Key metrics to show trends for
        key_metrics = ['spend_eur', 'revenue_eur', 'roas', 'cpa_eur', 'ctr_pct', 'conversion_rate_pct']
        
        for metric in key_metrics:
            if metric in trends:
                trend_data = trends[metric]
                direction = "📈" if trend_data['pct_change'] > 0 else "📉" if trend_data['pct_change'] < 0 else "➡️"
                report.append(f"{metric.replace('_', ' ').title()}: {direction} {trend_data['pct_change']:+.1f}%")
        report.append("")
    
    # Best and worst days
    if len(recent_data) > 1:
        best_roas_day = recent_data.loc[recent_data['roas'].idxmax()]
        worst_roas_day = recent_data.loc[recent_data['roas'].idxmin()]
        
        report.append("🏆 BEST & WORST PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Best ROAS Day: {best_roas_day['date'].strftime('%Y-%m-%d')} (ROAS: {best_roas_day['roas']:.2f})")
        report.append(f"Worst ROAS Day: {worst_roas_day['date'].strftime('%Y-%m-%d')} (ROAS: {worst_roas_day['roas']:.2f})")
        report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 30)
    
    if total_spend > 0:
        avg_roas = total_revenue / total_spend
        if avg_roas < 2.0:
            report.append("⚠️ Low ROAS - Consider optimizing targeting or creative")
        elif avg_roas > 4.0:
            report.append("✅ Strong ROAS - Consider scaling up budget")
        
        avg_cpa = total_spend / total_purchases if total_purchases > 0 else 0
        if avg_cpa > 50:
            report.append("⚠️ High CPA - Review audience targeting and creative relevance")
        
        avg_ctr = recent_data['ctr_pct'].mean()
        if avg_ctr < 1.0:
            report.append("⚠️ Low CTR - Test new creative concepts")
        elif avg_ctr > 3.0:
            report.append("✅ Strong CTR - Creative is performing well")
    
    report.append("")
    report.append("📊 Analysis completed at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return "\n".join(report)


def create_performance_charts(df: pd.DataFrame, output_dir: str = "charts") -> List[str]:
    """Create performance charts and save them."""
    if len(df) == 0:
        return []
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    chart_files = []
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Daily Spend and Revenue
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(df['date'], df['spend_eur'], marker='o', linewidth=2, label='Spend')
    ax1.plot(df['date'], df['revenue_eur'], marker='s', linewidth=2, label='Revenue')
    ax1.set_title('Daily Spend vs Revenue', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amount (€)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['date'], df['roas'], marker='o', linewidth=2, color='green')
    ax2.set_title('Daily ROAS', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ROAS')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'spend_revenue_roas.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    chart_files.append(chart_path)
    plt.close()
    
    # 2. Traffic Metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(df['date'], df['impressions'], marker='o', linewidth=2)
    ax1.set_title('Daily Impressions', fontweight='bold')
    ax1.set_ylabel('Impressions')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['date'], df['clicks'], marker='o', linewidth=2, color='orange')
    ax2.set_title('Daily Clicks', fontweight='bold')
    ax2.set_ylabel('Clicks')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(df['date'], df['ctr_pct'], marker='o', linewidth=2, color='red')
    ax3.set_title('Daily CTR (%)', fontweight='bold')
    ax3.set_ylabel('CTR %')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(df['date'], df['cpc_eur'], marker='o', linewidth=2, color='purple')
    ax4.set_title('Daily CPC (€)', fontweight='bold')
    ax4.set_ylabel('CPC €')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'traffic_metrics.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    chart_files.append(chart_path)
    plt.close()
    
    # 3. Conversion Funnel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Conversion rates over time
    ax1.plot(df['date'], df['conversion_rate_pct'], marker='o', linewidth=2, label='Conversion Rate')
    ax1.plot(df['date'], df['atc_rate_pct'], marker='s', linewidth=2, label='ATC Rate')
    ax1.plot(df['date'], df['ic_rate_pct'], marker='^', linewidth=2, label='IC Rate')
    ax1.set_title('Conversion Rates Over Time', fontweight='bold')
    ax1.set_ylabel('Rate (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Funnel rates
    ax2.plot(df['date'], df['atc_to_ic_rate_pct'], marker='o', linewidth=2, label='ATC to IC')
    ax2.plot(df['date'], df['ic_to_purchase_rate_pct'], marker='s', linewidth=2, label='IC to Purchase')
    ax2.set_title('Funnel Conversion Rates', fontweight='bold')
    ax2.set_ylabel('Rate (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'conversion_funnel.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    chart_files.append(chart_path)
    plt.close()
    
    return chart_files


def main():
    """Main function to analyze performance data."""
    parser = argparse.ArgumentParser(description='Analyze Meta Ads performance data')
    parser.add_argument('--csv-path', type=str, default='data/daily_performance_tracking.csv',
                       help='Path to the CSV file to analyze')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of recent days to focus on for trends')
    parser.add_argument('--charts', action='store_true',
                       help='Generate performance charts')
    parser.add_argument('--output-dir', type=str, default='charts',
                       help='Directory to save charts')
    
    args = parser.parse_args()
    
    print("📊 Loading performance data...")
    df = load_performance_data(args.csv_path)
    
    if len(df) == 0:
        print("❌ No data available for analysis.")
        return
    
    print(f"✅ Loaded {len(df)} days of performance data")
    
    # Generate performance report
    print("\n" + "="*60)
    report = generate_performance_report(df, args.days)
    print(report)
    print("="*60)
    
    # Generate charts if requested
    if args.charts:
        print(f"\n📈 Generating performance charts...")
        try:
            chart_files = create_performance_charts(df, args.output_dir)
            print(f"✅ Generated {len(chart_files)} charts:")
            for chart_file in chart_files:
                print(f"   📊 {chart_file}")
        except Exception as e:
            print(f"⚠️ Error generating charts: {e}")
    
    print(f"\n✅ Performance analysis completed!")


if __name__ == "__main__":
    main()
