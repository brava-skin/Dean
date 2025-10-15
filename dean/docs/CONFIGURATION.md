# Configuration Guide

This guide covers all configuration options for Dean automation system, including environment variables, YAML settings, and business rules.

## Environment Variables

### Required Variables

These variables must be set for the system to function:

```bash
# Meta API Credentials
FB_APP_ID=your_facebook_app_id
FB_APP_SECRET=your_facebook_app_secret
FB_ACCESS_TOKEN=your_long_lived_access_token
FB_AD_ACCOUNT_ID=act_your_account_id
FB_PIXEL_ID=your_pixel_id
FB_PAGE_ID=your_page_id

# Store Configuration
STORE_URL=https://your-store.com
```

### Optional Variables

```bash
# Instagram Integration
IG_ACTOR_ID=your_instagram_actor_id

# Slack Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_WEBHOOK_URL_ERRORS=https://hooks.slack.com/services/YOUR/ERROR/WEBHOOK

# Supabase Queue Management
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives

# Economics Configuration
BREAKEVEN_CPA=34
COGS_PER_PURCHASE=15
USD_EUR_RATE=0.92

# Account Configuration
ACCOUNT_TIMEZONE=Europe/Amsterdam
ACCOUNT_CURRENCY=EUR
ACCOUNT_CURRENCY_SYMBOL=€

# Execution Control
DRY_RUN=false
MODE=production
```

### Advanced Variables

```bash
# Meta API Configuration
FB_API_VERSION=19.0
META_RETRY_MAX=4
META_BACKOFF_BASE=0.4
META_TIMEOUT=30
META_WRITE_COOLDOWN_SEC=5

# Budget Limits
BUDGET_MIN=5
BUDGET_MAX=50000
BUDGET_MAX_STEP_PCT=200

# Circuit Breaker
META_CB_FAILS=5
META_CB_RESET_SEC=120

# Creative Compliance
FORBIDDEN_TERMS=cures,miracle,guaranteed
```

## Settings Configuration (`config/settings.yaml`)

### Account Configuration

```yaml
# Account Information
ids:
  testing_campaign_id: "120231838265440160"
  testing_adset_id: "120231838265460160"
  validation_campaign_id: "120231838417260160"
  scaling_campaign_id: "120231838441470160"

# Timezone Configuration
account_timezone: "Europe/Amsterdam"
notifications_timezone: "Europe/Amsterdam"
```

### Testing Stage Configuration

```yaml
testing:
  # Budget Settings
  daily_budget_eur: 50
  daily_budget_usd: 50
  
  # Ad Management
  keep_ads_live: 4
  max_active_ads: 4
  
  # Performance Settings
  optimization_goal: "PURCHASE"
  attribution: "7d_click_1d_view"
  bid_strategy: "lowest_cost"
  dynamic_creative: false
  
  # Minimum Requirements
  minimums:
    min_impressions: 300
    min_clicks: 0
    min_spend_eur: 10
    min_spend_usd: 10
  
  # Fairness Rules
  fairness:
    min_spend_before_kill_eur: 20
    min_spend_before_kill_usd: 20
    min_runtime_hours: 12
  
  # Queue Management
  queue_rotation:
    enable: true
    refill_on_kill: true
    max_launches_per_day: 8
  
  # Naming
  naming_prefix: "[TEST]"
  
  # Placements
  placements:
    publisher_platforms: ["facebook", "instagram"]
    device_platforms: ["mobile", "desktop"]
    facebook_positions: ["feed", "video_feeds"]
    instagram_positions: ["feed", "story", "reels"]
```

### Validation Stage Configuration

```yaml
validation:
  # Budget Settings
  adset_budget_eur: 40
  adset_budget_usd: 40
  
  # Duration Settings
  min_days: 2
  max_days: 3
  total_spend_cap_eur: 120
  total_spend_cap_usd: 120
  
  # Performance Settings
  optimization_goal: "PURCHASE"
  attribution: "7d_click_1d_view"
  bid_strategy: "lowest_cost"
  dynamic_creative: false
  
  # Minimum Requirements
  minimums:
    min_impressions: 600
    min_clicks: 30
    min_spend_eur: 40
    min_spend_usd: 40
  
  # Promotion Settings
  promotion:
    pause_validation_on_promotion: true
  
  # Naming
  naming_prefix: "[VALID]"
```

### Scaling Stage Configuration

```yaml
scaling:
  # Budget Settings
  adset_start_budget_eur: 100
  adset_start_budget_usd: 100
  
  # CBO Settings
  allow_cbo: true
  cbo_entry_budget_eur: 400
  cbo_entry_budget_usd: 400
  cbo_increment_pct: [50, 100]
  
  # Performance Settings
  optimization_goal: "PURCHASE"
  attribution: "7d_click_1d_view"
  bid_strategy: "lowest_cost"
  dynamic_creative: true
  
  # Budget Management
  budget:
    min_eur: 10
    max_eur: 5000
    min_usd: 10
    max_usd: 5000
    max_step_pct: 100
    cooldown_hours_after_scale: 24
  
  # Hysteresis Protection
  hysteresis:
    roas_down_band: 1.7
    cpa_up_band: 33
  
  # Reinvestment Strategy
  reinvestment:
    enabled: true
    strategy: "proportional_to_roas"
    allocate_pct_of_freed: 50
    min_increment_eur: 10
    min_increment_usd: 10
  
  # Winner Refresh
  refresh_winner:
    enabled: true
    after_good_days: 7
    cpa_max: 22
    roas_min: 3.0
  
  # Naming
  naming_prefix: "[SCALE]"
```

### Guardrails Configuration

```yaml
guardrails:
  # Account-Level Protection
  account_cpa_pause_factor: 1.5
  account_cpa_window_days: 2
  no_purchases_tripwire_hours: 48
  
  # Daily Spending Limits
  global_daily_cap_eur: 3000
  global_daily_cap_usd: 3000
  
  # Spend Velocity Monitoring
  spend_velocity:
    compare_24h_vs_prev_24h: true
    compare_48h_vs_prev_48h: true
    warn_24h_growth_x: 1.5
    warn_48h_growth_x: 2.5
    freeze_scaling_if_exceeds: true
  
  # Scaling Protection
  scaling_freeze_days_on_worsen: 3
  staging_relax_thresholds: true
```

### Queue Configuration

```yaml
queue:
  # Schema Definition
  schema:
    required_columns:
      - video_id
    optional_columns:
      - creative_id
      - name
      - avatar
      - visual_style
      - script
      - video_url
      - thumbnail_url
      - page_id
      - utm_params
      - cogs_override_eur
      - tags
  
  # Deduplication
  dedupe:
    by: ["creative_id", "name"]
  
  # Prioritization
  prioritization:
    strategy: "fifo"
    tag_weights: {}
  
  # Validation
  validation:
    check_media_exists: true
    min_duration_sec: 5
    max_duration_sec: 120
```

### Notifications Configuration

```yaml
notifications:
  slack:
    enabled: true
    channels:
      default_env: "SLACK_WEBHOOK_URL"
      errors_env: "SLACK_WEBHOOK_URL_ERRORS"
    route:
      info: "default"
      warn: "default"
      error: "errors"
    digest:
      enabled: true
      time_local: "08:30"
      include_tables: true
      include_stage_summaries: true
    templates:
      kill: "🛑 [{stage}] Killed {name} — {reason} (CPA {cpa}, ROAS {roas})"
      promote: "🚀 [{from}→{to}] {name} — {reason} (start €{budget}/d)"
      scale: "⬆️ [SCALE] {name} +{pct}% → €{new_budget}"
      duplicate: "🧬 [SCALE] {name} ×{count} (PAUSED)"
```

## Rules Configuration (`config/rules.yaml`)

### Performance Thresholds

```yaml
thresholds:
  # Click-Through Rate Thresholds
  ctr:
    testing_floor: 0.008
    validation_floor: 0.010
  
  # Cost Per Acquisition (EUR)
  cpa:
    testing_max: 36
    validation_max: 28
    scaling_kill_max: 40
  
  # Return on Ad Spend
  roas:
    testing_min: 1.5
    validation_min: 1.8
    scaling_kill_min: 1.2
  
  # Purchase on Ad Spend
  poas:
    validation_min: 0.0
    scaling_min: 0.2
  
  # Average Order Value
  aov:
    validation_min: null
```

### Testing Rules

```yaml
testing:
  # Kill Rules
  kill:
    - {type: "spend_no_purchase", spend_gte: 45}
    - {type: "ctr_below", ctr_lt: 0.008, spend_gte: 40}
    - {type: "spend_no_atc", spend_gte: 50}
    - {type: "spend_multi_atc_no_purchase", spend_gte: 70, atc_gte: 2}
    - {type: "roas_below_after_days", roas_lt: 1.5, window: "day_3"}
    - {type: "cpa_gte_over_days", cpa_gte: 38, days: 3}
    - {type: "spend_over_2x_price_no_purchase"}
  
  # Advance Rules
  advance:
    operator: "all"
    rules:
      - {type: "purchase_gte", purchases_gte: 1}
      - {type: "cpa_lte", cpa_lte: 36}
      - {type: "ctr_gte", ctr_gte: 0.008}
```

### Validation Rules

```yaml
validation:
  # Kill Rules
  kill:
    - {type: "spend_no_purchase", spend_gte: 40}
    - {type: "cpa_gte", cpa_gte: 32}
    - {type: "ctr_below", ctr_lt: 0.008, spend_gte: 40}
    - {type: "spend_multi_atc_no_purchase", spend_gte: 80, atc_gte: 2}
    - {type: "roas_below_after_days", roas_lt: 1.5, window: "day_3"}
    - {type: "spend_over_2x_price_no_purchase"}
  
  # Advance Rules
  advance:
    operator: "all"
    rules:
      - {type: "purchase_gte", purchases_gte: 2}
      - {type: "cpa_lte", cpa_lte: 28}
      - {type: "roas_gte", roas_gte: 1.8}
      - {type: "ctr_gte", ctr_gte: 0.01}
  
  # Strictness Requirements
  strictness:
    min_days_with_purchase: 2
```

### Scaling Rules

```yaml
scaling:
  # Kill Rules
  kill:
    - {type: "cpa_gte_consecutive_days", cpa_gte: 40, days: 2}
    - {type: "roas_lt_over_days", roas_lt: 1.2, days: 3}
    - {type: "spend_no_purchase", spend_gte: 150}
  
  # Scale Up Rules
  scale_up:
    cooldown_hours: 24
    hysteresis:
      roas_down_band: 1.7
      cpa_up_band: 33
    steps:
      - {type: "cpa_lte_and_roas_gte_over_days", cpa_lte: 27, roas_gte: 2.0, days: 2, budget_increase_pct: [50, 100]}
      - {type: "cpa_lte_and_roas_gte_over_days", cpa_lte: 22, roas_gte: 3.0, days: 2, budget_increase_pct: [100, 200]}
  
  # Duplication Rules
  duplicate_on_fire:
    max_duplicates_per_24h: 3
    rules:
      - {type: "purchases_in_day", purchases_gte: 5, cpa_lte: 27, duplicates: 2, budget_each: 150}
```

## Advanced Configuration

### Creative Compliance

```yaml
creative_compliance:
  require_fields: ["primary_text", "headline"]
  max_lengths:
    primary_text: 300
    headline: 60
    description: 150
  forbid_terms: ["cures", "miracle", "guaranteed"]
  auto_truncate: true
  require_utm: false
```

### Naming Conventions

```yaml
naming:
  enforce: true
  auto_prefix_if_missing: true
  patterns:
    campaign: ["^\\[(TEST|VALID|SCALE|SCALE-CBO)\\]\\s+Brava\\s+—\\s+(ABO|CBO)\\s+—\\s+US Men$"]
    adset: ["^\\[(TEST|VALID|SCALE)\\]\\s+.+$"]
    ad: ["^\\[(TEST|VALID|SCALE)\\]\\s+.+$"]
  warn_on_mismatch: true
```

### Data Quality Monitoring

```yaml
data_quality:
  missing_actions_alert:
    enabled: true
    min_spend_for_alert_eur: 20
    min_spend_for_alert_usd: 20
  tracking_drop_alert:
    enabled: true
    drop_pct_vs_7d_avg: 60
```

### Execution Control

```yaml
execution:
  dry_run_env: "DRY_RUN"
  rate_limits:
    retry_max: 5
    backoff_base_seconds: 0.6
    respect_retry_after: true
  pagination:
    insights_page_limit: 500
    max_pages: 50
  concurrency:
    writes_max_parallel: 3
    reads_max_parallel: 5
```

## Configuration Validation

### Environment Check

```bash
# Check required environment variables
python -c "
import os
required = ['FB_APP_ID', 'FB_APP_SECRET', 'FB_ACCESS_TOKEN', 'FB_AD_ACCOUNT_ID']
missing = [var for var in required if not os.getenv(var)]
if missing:
    print(f'Missing variables: {missing}')
else:
    print('All required variables set')
"
```

### Settings Validation

```bash
# Validate YAML configuration
python -c "
import yaml
with open('config/settings.yaml') as f:
    settings = yaml.safe_load(f)
print('Settings file is valid YAML')
"
```

### Rules Validation

```bash
# Validate rules configuration
python -c "
import yaml
with open('config/rules.yaml') as f:
    rules = yaml.safe_load(f)
print('Rules file is valid YAML')
"
```

## Best Practices

### Security

- Store sensitive credentials in environment variables
- Use service accounts with minimal required permissions
- Rotate access tokens regularly
- Monitor API usage and costs

### Performance

- Set appropriate budget limits for your business model
- Configure realistic performance thresholds
- Use staging mode for testing configuration changes
- Monitor system health and performance

### Maintenance

- Regular backup of configuration files
- Version control for configuration changes
- Document configuration changes
- Test configuration changes in staging first

## Troubleshooting Configuration

### Common Issues

**1. Missing Environment Variables**
```bash
# Check if variables are loaded
python -c "import os; print(os.getenv('FB_APP_ID'))"
```

**2. Invalid YAML Syntax**
```bash
# Validate YAML files
python -c "import yaml; yaml.safe_load(open('config/settings.yaml'))"
```

**3. Configuration Conflicts**
```bash
# Check for conflicting settings
python src/main.py --dry-run --explain
```

**4. API Permission Issues**
```bash
# Test API access
python -c "
from src.meta_client import MetaClient, AccountAuth
# Test API connection
"
```

### Configuration Tips

- Start with conservative thresholds and adjust based on performance
- Use staging mode to test configuration changes
- Monitor system logs for configuration-related errors
- Keep configuration files in version control
- Document any custom configuration changes
