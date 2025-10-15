# Installation Guide

This guide will walk you through setting up Dean for automated Facebook/Meta advertising management.

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB free space for logs and data

### Account Requirements

- **Meta Business Account**: With advertising permissions
- **Facebook Developer Account**: For API access
- **Slack Workspace**: For notifications (optional but recommended)
- **Supabase Account**: For queue management (optional)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd dean
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```bash
# Meta API Configuration
FB_APP_ID=your_facebook_app_id
FB_APP_SECRET=your_facebook_app_secret
FB_ACCESS_TOKEN=your_long_lived_access_token
FB_AD_ACCOUNT_ID=act_your_account_id
FB_PIXEL_ID=your_pixel_id
FB_PAGE_ID=your_page_id

# Store Configuration
STORE_URL=https://your-store.com

# Optional: Instagram
IG_ACTOR_ID=your_instagram_actor_id

# Slack Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_WEBHOOK_URL_ERRORS=https://hooks.slack.com/services/YOUR/ERROR/WEBHOOK

# Optional: Supabase Queue Management
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives

# Optional: Economics Configuration
BREAKEVEN_CPA=34
COGS_PER_PURCHASE=15
USD_EUR_RATE=0.92

# Optional: Advanced Configuration
ACCOUNT_TIMEZONE=Europe/Amsterdam
ACCOUNT_CURRENCY=EUR
DRY_RUN=false
```

### 5. Meta API Setup

#### Create Facebook App

1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create a new app or use existing one
3. Add "Marketing API" product
4. Generate app secret

#### Generate Access Token

1. Go to Graph API Explorer
2. Select your app
3. Generate long-lived access token (60 days)
4. Add required permissions:
   - `ads_management`
   - `ads_read`
   - `business_management`

#### Get Account Information

```bash
# Get your ad account ID
curl -G \
  -d "access_token=YOUR_ACCESS_TOKEN" \
  "https://graph.facebook.com/v19.0/me/adaccounts"

# Get pixel ID
curl -G \
  -d "access_token=YOUR_ACCESS_TOKEN" \
  "https://graph.facebook.com/v19.0/act_YOUR_ACCOUNT_ID/adspixels"
```

### 6. Slack Integration (Optional)

#### Create Slack App

1. Go to [Slack API](https://api.slack.com/apps)
2. Create new app
3. Enable Incoming Webhooks
4. Create webhook URLs for notifications

#### Configure Webhooks

- **Default Channel**: General notifications
- **Error Channel**: Error alerts and warnings

### 7. Supabase Setup (Optional)

#### Create Supabase Project

1. Go to [Supabase](https://supabase.com/)
2. Create new project
3. Get project URL and service role key

#### Create Queue Table

```sql
CREATE TABLE meta_creatives (
  id SERIAL PRIMARY KEY,
  video_id TEXT,
  filename TEXT,
  avatar TEXT,
  visual_style TEXT,
  script TEXT,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Add indexes for performance
CREATE INDEX idx_meta_creatives_status ON meta_creatives(status);
CREATE INDEX idx_meta_creatives_video_id ON meta_creatives(video_id);
```

### 8. Configuration Files

#### Settings Configuration

Edit `config/settings.yaml` with your account details:

```yaml
# Account IDs (replace with your actual IDs)
ids:
  testing_campaign_id: "120231838265440160"
  testing_adset_id: "120231838265460160"
  validation_campaign_id: "120231838417260160"
  scaling_campaign_id: "120231838441470160"

# Budget Configuration
testing:
  daily_budget_eur: 50
  daily_budget_usd: 50

validation:
  adset_budget_eur: 40
  adset_budget_usd: 40

scaling:
  adset_start_budget_eur: 100
  adset_start_budget_usd: 100
```

#### Rules Configuration

Edit `config/rules.yaml` with your business rules:

```yaml
# Performance Thresholds
thresholds:
  cpa:
    testing_max: 36
    validation_max: 28
    scaling_kill_max: 40
  roas:
    testing_min: 1.5
    validation_min: 1.8
    scaling_kill_min: 1.2

# Testing Rules
testing:
  kill:
    - {type: "spend_no_purchase", spend_gte: 45}
    - {type: "ctr_below", ctr_lt: 0.008, spend_gte: 40}
    - {type: "cpa_gte_over_days", cpa_gte: 38, days: 3}
```

### 9. Test Installation

#### Basic Health Check

```bash
# Test configuration
python src/main.py --dry-run --explain

# Test specific stage
python src/main.py --stage testing --dry-run
```

#### Verify API Access

```bash
# Test Meta API connection
python -c "
from src.meta_client import MetaClient, AccountAuth
from src.storage import Store

# Test basic API access
store = Store('data/test.sqlite')
auth = AccountAuth(
    account_id='your_account_id',
    access_token='your_token',
    app_id='your_app_id',
    app_secret='your_secret'
)
client = MetaClient([auth], store=store, dry_run=True)
print('API connection successful!')
"
```

#### Test Slack Notifications

```bash
# Test Slack webhook
python -c "
from src.slack import notify
notify('🧪 Test notification from Dean automation')
"
```

### 10. Production Setup

#### Create Data Directories

```bash
mkdir -p data/digests
mkdir -p data/snapshots
mkdir -p logs
```

#### Set Up Logging

```bash
# Create log rotation script
cat > rotate_logs.sh << 'EOF'
#!/bin/bash
# Rotate logs weekly
find logs/ -name "*.log" -mtime +7 -delete
find data/digests/ -name "*.jsonl" -mtime +30 -delete
EOF

chmod +x rotate_logs.sh
```

#### Schedule Automation

```bash
# Add to crontab for regular execution
# Run every 2 hours
0 */2 * * * cd /path/to/dean && python src/main.py --profile production

# Run daily at 9 AM
0 9 * * * cd /path/to/dean && python src/main.py --stage testing
```

## Verification Steps

### 1. Check Environment Variables

```bash
python -c "
import os
required = ['FB_APP_ID', 'FB_APP_SECRET', 'FB_ACCESS_TOKEN', 'FB_AD_ACCOUNT_ID']
missing = [var for var in required if not os.getenv(var)]
if missing:
    print(f'Missing: {missing}')
else:
    print('All required environment variables set')
"
```

### 2. Test Database Connection

```bash
python -c "
from src.storage import Store
store = Store('data/test.sqlite')
store.log('test', 'test', 'TEST', 'info', 'test', 'Installation test')
print('Database connection successful')
"
```

### 3. Test Meta API

```bash
python -c "
from src.meta_client import MetaClient, AccountAuth, ClientConfig
from src.storage import Store
import os

store = Store('data/test.sqlite')
auth = AccountAuth(
    account_id=os.getenv('FB_AD_ACCOUNT_ID'),
    access_token=os.getenv('FB_ACCESS_TOKEN'),
    app_id=os.getenv('FB_APP_ID'),
    app_secret=os.getenv('FB_APP_SECRET')
)
client = MetaClient([auth], ClientConfig(), store=store, dry_run=True)

# Test basic API call
try:
    insights = client.get_ad_insights(level='ad', fields=['spend'], paginate=False)
    print(f'API test successful: {len(insights)} ads found')
except Exception as e:
    print(f'API test failed: {e}')
"
```

### 4. Test Slack Integration

```bash
python -c "
from src.slack import notify
try:
    notify('🧪 Dean installation test successful!')
    print('Slack integration working')
except Exception as e:
    print(f'Slack test failed: {e}')
"
```

## Troubleshooting Installation

### Common Issues

**1. Python Version Issues**
```bash
# Check Python version
python --version
# Should be 3.9 or higher

# If using older version, install Python 3.9+
```

**2. Missing Dependencies**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check for missing packages
pip check
```

**3. Environment Variable Issues**
```bash
# Check if .env file is loaded
python -c "import os; print(os.getenv('FB_APP_ID'))"

# If None, check .env file location and format
```

**4. API Permission Issues**
```bash
# Test API permissions
python -c "
from src.meta_client import MetaClient, AccountAuth
# Check if you can access ad account
"
```

**5. Database Issues**
```bash
# Check SQLite installation
python -c "import sqlite3; print('SQLite OK')"

# Check database permissions
touch data/test.sqlite
```

### Getting Help

If you encounter issues during installation:

1. Check the troubleshooting section in the main README
2. Verify all environment variables are set correctly
3. Test API access with Meta's Graph API Explorer
4. Check Slack webhook URLs are valid
5. Ensure all required permissions are granted

## Next Steps

After successful installation:

1. **Configure your first campaign** in `config/settings.yaml`
2. **Set up your creative queue** (Supabase or CSV)
3. **Test with dry-run mode** before going live
4. **Schedule regular execution** for production use
5. **Monitor Slack notifications** for system health

## Security Considerations

- Store credentials securely (use environment variables)
- Rotate access tokens regularly
- Use service accounts with minimal permissions
- Monitor API usage and costs
- Implement proper backup procedures
