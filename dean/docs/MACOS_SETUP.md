# macOS Setup Guide

This guide provides macOS-specific setup instructions and optimizations for Dean.

## 🍎 macOS-Specific Features

### Automated Setup Script

Use the included setup script for a streamlined installation:

```bash
chmod +x scripts/setup_macos.sh
./scripts/setup_macos.sh
```

This script will:
- Install Homebrew (if not present)
- Install Python 3.11 via Homebrew
- Set up virtual environment
- Install all dependencies
- Create necessary directories
- Set up log rotation
- Create launchd service template

### Homebrew Integration

The setup script uses Homebrew for package management, which provides:
- Latest Python versions
- Better dependency management
- Easy updates and maintenance

### Launchd Service (Recommended)

Instead of cron, macOS users should use launchd for more reliable service management:

```bash
# After running setup script, configure the service
cp ~/Library/LaunchAgents/com.dean.automation.plist.template \
   ~/Library/LaunchAgents/com.dean.automation.plist

# Edit the plist file to match your paths
nano ~/Library/LaunchAgents/com.dean.automation.plist

# Load the service
launchctl load ~/Library/LaunchAgents/com.dean.automation.plist

# Check status
launchctl list | grep dean

# Unload if needed
launchctl unload ~/Library/LaunchAgents/com.dean.automation.plist
```

### macOS-Specific Optimizations

#### Terminal Compatibility

The system automatically handles Unicode emoji display in Terminal.app:
- Emojis are converted to text equivalents for better compatibility
- Works with both Terminal.app and iTerm2
- Supports dark/light mode themes

#### File System

- Uses macOS-native file paths (`/` separators)
- Respects macOS file permissions
- Optimized for APFS (Apple File System)

#### Performance

- Leverages macOS's native SQLite implementation
- Optimized for Apple Silicon (M1/M2) processors
- Uses native threading for better performance

## 🔧 Configuration

### Environment Variables

Create your `.env` file in the project root:

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

### Settings Configuration

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

## 🚀 Running Dean

### Development Mode

```bash
# Activate virtual environment
source venv/bin/activate

# Run in dry-run mode
python src/main.py --dry-run --explain

# Run specific stage
python src/main.py --stage testing --dry-run
```

### Production Mode

```bash
# Run with production profile
python src/main.py --profile production

# Run with launchd service (recommended)
# Service will start automatically after setup
```

## 📊 Monitoring

### Log Files

Logs are stored in the `logs/` directory:
- `automation.log`: General automation logs
- `automation_error.log`: Error logs
- Rotated weekly automatically

### Console App

View logs in real-time:
```bash
# Follow main log
tail -f logs/automation.log

# Follow error log
tail -f logs/automation_error.log

# View all logs
tail -f logs/*.log
```

### Activity Monitor

Monitor system resources:
- CPU usage for Python processes
- Memory usage for automation tasks
- Network activity for API calls

## 🔍 Troubleshooting

### Common Issues

**Python Path Issues:**
```bash
# Ensure you're using the correct Python
which python3
# Should show: /opt/homebrew/bin/python3 (Apple Silicon) or /usr/local/bin/python3 (Intel)

# Activate virtual environment
source venv/bin/activate
which python
# Should show: /path/to/dean/venv/bin/python
```

**Permission Issues:**
```bash
# Fix file permissions
chmod +x scripts/setup_macos.sh
chmod +x rotate_logs.sh

# Fix directory permissions
chmod -R 755 data/
chmod -R 755 logs/
```

**Launchd Service Issues:**
```bash
# Check service status
launchctl list | grep dean

# View service logs
log show --predicate 'process == "dean"' --last 1h

# Reload service
launchctl unload ~/Library/LaunchAgents/com.dean.automation.plist
launchctl load ~/Library/LaunchAgents/com.dean.automation.plist
```

**API Connection Issues:**
```bash
# Test API connection
python -c "
from src.meta_client import MetaClient, AccountAuth
from src.storage import Store
import os

store = Store('data/test.sqlite')
auth = AccountAuth(
    account_id=os.getenv('FB_AD_ACCOUNT_ID'),
    access_token=os.getenv('FB_ACCESS_TOKEN'),
    app_id=os.getenv('FB_APP_ID'),
    app_secret=os.getenv('FB_APP_SECRET')
)
client = MetaClient([auth], store=store, dry_run=True)
print('✅ API connection successful!')
"
```

## 🔄 Updates

### Updating Dean

```bash
# Pull latest changes
git pull origin main

# Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Restart service
launchctl unload ~/Library/LaunchAgents/com.dean.automation.plist
launchctl load ~/Library/LaunchAgents/com.dean.automation.plist
```

### Updating macOS

After macOS updates:
1. Verify Python installation: `python3 --version`
2. Reinstall dependencies if needed: `pip install -r requirements.txt`
3. Restart launchd service
4. Test automation: `python src/main.py --dry-run`

## 📚 Additional Resources

- [Homebrew Documentation](https://docs.brew.sh/)
- [Launchd Documentation](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/)
- [Python on macOS](https://docs.python.org/3/using/mac.html)
- [Meta Marketing API](https://developers.facebook.com/docs/marketing-api/)

## 🤝 Support

For macOS-specific issues:
1. Check this guide first
2. Verify Homebrew and Python installation
3. Test with dry-run mode
4. Check launchd service status
5. Review log files for errors
