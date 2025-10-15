# Environment Setup Guide

## Required Environment Variables

Create a `.env` file in the `dean/` directory with the following variables:

```bash
# Meta API Configuration
FB_APP_ID=your_app_id_here
FB_APP_SECRET=your_app_secret_here
FB_ACCESS_TOKEN=your_access_token_here
FB_AD_ACCOUNT_ID=your_ad_account_id_here
FB_PIXEL_ID=your_pixel_id_here
FB_PAGE_ID=your_page_id_here

# Store Configuration
STORE_URL=https://your-store.com
IG_ACTOR_ID=your_instagram_actor_id_here

# Slack Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_WEBHOOK_URL_ERRORS=https://hooks.slack.com/services/YOUR/ERRORS/WEBHOOK

# Supabase Configuration (REQUIRED - main queue source)
SUPABASE_URL=your_supabase_url_here
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key_here
SUPABASE_TABLE=meta_creatives

# Economics
BREAKEVEN_CPA=25.0
COGS_PER_PURCHASE=8.0
USD_EUR_RATE=0.85

# Execution
DRY_RUN=false
```

## How to Get These Values

### Meta API Credentials
1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create a new app or use existing one
3. Get App ID and App Secret from App Settings
4. Generate a long-lived access token with required permissions
5. Get Ad Account ID from Meta Business Manager
6. Get Pixel ID from Meta Business Manager
7. Get Page ID from your Facebook page

### Slack Webhooks
1. Go to your Slack workspace
2. Create a new app or use existing one
3. Enable Incoming Webhooks
4. Create webhook URLs for notifications

### Supabase (Required)
1. Create a Supabase project at [supabase.com](https://supabase.com)
2. Get the project URL and service role key from Settings > API
3. Run the setup script: `python setup_supabase.py`
4. Follow the instructions to create the table with the correct schema

## Testing Without Full Setup

If you want to test the system without all credentials:

1. Set `DRY_RUN=true` in your `.env` file
2. The system will run in simulation mode
3. No actual API calls will be made
4. You can see the logic flow without spending money

## Queue Configuration

The system uses **Supabase as the primary queue source**. Make sure to:
1. Set up your Supabase project
2. Configure the environment variables
3. Run `python setup_supabase.py` to create the table schema
4. Add creative data to your Supabase table

## Copy Bank

The copy bank is already configured in `data/copy_bank.json` and will be used automatically.
