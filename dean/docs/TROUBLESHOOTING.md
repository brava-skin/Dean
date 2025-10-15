# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Dean automation system.

## Quick Diagnostics

### System Health Check

```bash
# Check system health
python -c "
from src.main import health_check
from src.storage import Store
from src.meta_client import MetaClient, AccountAuth, ClientConfig
import os

store = Store('data/state.sqlite')
auth = AccountAuth(
    account_id=os.getenv('FB_AD_ACCOUNT_ID'),
    access_token=os.getenv('FB_ACCESS_TOKEN'),
    app_id=os.getenv('FB_APP_ID'),
    app_secret=os.getenv('FB_APP_SECRET')
)
client = MetaClient([auth], ClientConfig(), store=store, dry_run=True)

health = health_check(store, client)
print(f'Health: {health}')
"
```

### Configuration Validation

```bash
# Validate configuration files
python -c "
import yaml
try:
    with open('config/settings.yaml') as f:
        settings = yaml.safe_load(f)
    print('Settings: OK')
except Exception as e:
    print(f'Settings Error: {e}')

try:
    with open('config/rules.yaml') as f:
        rules = yaml.safe_load(f)
    print('Rules: OK')
except Exception as e:
    print(f'Rules Error: {e}')
"
```

### Environment Check

```bash
# Check required environment variables
python -c "
import os
required = ['FB_APP_ID', 'FB_APP_SECRET', 'FB_ACCESS_TOKEN', 'FB_AD_ACCOUNT_ID']
missing = [var for var in required if not os.getenv(var)]
if missing:
    print(f'Missing: {missing}')
else:
    print('Environment: OK')
"
```

## Common Issues

### 1. Installation Issues

#### Python Version Problems

**Symptoms:**
- Import errors
- Syntax errors
- Module not found errors

**Solutions:**
```bash
# Check Python version
python --version
# Should be 3.9 or higher

# If using older version, install Python 3.9+
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.9 python3.9-venv

# On macOS:
brew install python@3.9

# On Windows:
# Download from python.org
```

#### Missing Dependencies

**Symptoms:**
- `ModuleNotFoundError` when running
- Import errors for specific packages

**Solutions:**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check for missing packages
pip check

# Install specific missing package
pip install package_name

# If using virtual environment, ensure it's activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### Permission Errors

**Symptoms:**
- Permission denied errors
- Cannot create files/directories

**Solutions:**
```bash
# Check file permissions
ls -la data/
chmod 755 data/
chmod 644 config/*.yaml

# Create data directory if missing
mkdir -p data/digests
mkdir -p data/snapshots
mkdir -p logs
```

### 2. Configuration Issues

#### Missing Environment Variables

**Symptoms:**
- "Missing required environment variables" error
- API connection failures

**Solutions:**
```bash
# Check if .env file exists and is readable
ls -la .env
cat .env

# Verify variables are loaded
python -c "
import os
print('FB_APP_ID:', os.getenv('FB_APP_ID'))
print('FB_ACCESS_TOKEN:', os.getenv('FB_ACCESS_TOKEN')[:10] + '...' if os.getenv('FB_ACCESS_TOKEN') else None)
"

# If variables not loaded, check .env file format
# Should be: VARIABLE_NAME=value
# No spaces around =
# No quotes unless needed
```

#### Invalid YAML Configuration

**Symptoms:**
- YAML parsing errors
- Configuration not loading

**Solutions:**
```bash
# Validate YAML syntax
python -c "
import yaml
try:
    with open('config/settings.yaml') as f:
        yaml.safe_load(f)
    print('Settings YAML: OK')
except yaml.YAMLError as e:
    print(f'YAML Error: {e}')
"

# Check for common YAML issues:
# - Proper indentation (use spaces, not tabs)
# - Correct syntax for lists and dictionaries
# - No trailing commas
# - Proper string quoting
```

#### Account ID Issues

**Symptoms:**
- "Invalid account ID" errors
- Campaign/adset not found

**Solutions:**
```bash
# Verify account ID format
# Should be: act_123456789
echo $FB_AD_ACCOUNT_ID

# Test API access
python -c "
from src.meta_client import MetaClient, AccountAuth, ClientConfig
import os

auth = AccountAuth(
    account_id=os.getenv('FB_AD_ACCOUNT_ID'),
    access_token=os.getenv('FB_ACCESS_TOKEN'),
    app_id=os.getenv('FB_APP_ID'),
    app_secret=os.getenv('FB_APP_SECRET')
)
client = MetaClient([auth], ClientConfig(), dry_run=True)

try:
    # Test basic API call
    print('API connection successful')
except Exception as e:
    print(f'API Error: {e}')
"
```

### 3. API Issues

#### Rate Limiting

**Symptoms:**
- "Rate limit exceeded" errors
- API calls failing with 429 status

**Solutions:**
```bash
# Increase cooldown between API calls
export META_WRITE_COOLDOWN_SEC=30

# Reduce concurrent operations
export META_RETRY_MAX=3
export META_BACKOFF_BASE=1.0

# Check current rate limit settings
python -c "
import os
print('Cooldown:', os.getenv('META_WRITE_COOLDOWN_SEC', '5'))
print('Retry Max:', os.getenv('META_RETRY_MAX', '4'))
"
```

#### Authentication Failures

**Symptoms:**
- "Invalid access token" errors
- 401 Unauthorized responses

**Solutions:**
```bash
# Check token validity
python -c "
import requests
import os

token = os.getenv('FB_ACCESS_TOKEN')
if token:
    response = requests.get(f'https://graph.facebook.com/v19.0/me?access_token={token}')
    if response.status_code == 200:
        print('Token valid')
    else:
        print(f'Token invalid: {response.status_code}')
else:
    print('No token found')
"

# Generate new long-lived token
# Go to Graph API Explorer
# Select your app
# Generate long-lived token (60 days)
```

#### Permission Errors

**Symptoms:**
- "Insufficient permissions" errors
- Cannot access campaigns/adsets

**Solutions:**
```bash
# Check required permissions
# Go to Facebook Developers
# App Settings > Permissions
# Ensure these are granted:
# - ads_management
# - ads_read
# - business_management

# Test specific permissions
python -c "
from src.meta_client import MetaClient, AccountAuth, ClientConfig
import os

auth = AccountAuth(
    account_id=os.getenv('FB_AD_ACCOUNT_ID'),
    access_token=os.getenv('FB_ACCESS_TOKEN'),
    app_id=os.getenv('FB_APP_ID'),
    app_secret=os.getenv('FB_APP_SECRET')
)
client = MetaClient([auth], ClientConfig(), dry_run=True)

try:
    # Test campaign access
    campaigns = client.get_campaigns()
    print(f'Found {len(campaigns)} campaigns')
except Exception as e:
    print(f'Permission Error: {e}')
"
```

### 4. Database Issues

#### SQLite Lock Errors

**Symptoms:**
- "Database is locked" errors
- Concurrent access issues

**Solutions:**
```bash
# Check for concurrent processes
ps aux | grep python

# Kill any stuck processes
pkill -f "python src/main.py"

# Check database file permissions
ls -la data/state.sqlite
chmod 644 data/state.sqlite

# Test database access
python -c "
from src.storage import Store
store = Store('data/state.sqlite')
store.log('test', 'test', 'TEST', 'info', 'test', 'Database test')
print('Database access: OK')
"
```

#### Database Corruption

**Symptoms:**
- Database errors
- Data not persisting

**Solutions:**
```bash
# Check database integrity
sqlite3 data/state.sqlite "PRAGMA integrity_check;"

# If corrupted, backup and recreate
cp data/state.sqlite data/state.sqlite.backup
rm data/state.sqlite

# Test database recreation
python -c "
from src.storage import Store
store = Store('data/state.sqlite')
store.log('test', 'test', 'TEST', 'info', 'test', 'Recreated database')
print('Database recreated: OK')
"
```

### 5. Slack Integration Issues

#### Webhook Failures

**Symptoms:**
- Slack notifications not working
- Webhook errors

**Solutions:**
```bash
# Test Slack webhook
python -c "
from src.slack import notify
try:
    notify('🧪 Test notification from Dean')
    print('Slack: OK')
except Exception as e:
    print(f'Slack Error: {e}')
"

# Check webhook URL format
# Should be: https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
echo $SLACK_WEBHOOK_URL

# Verify webhook is active
# Go to Slack App Settings
# Incoming Webhooks > Check if enabled
```

#### Channel Issues

**Symptoms:**
- Notifications not appearing
- Wrong channel

**Solutions:**
```bash
# Check webhook configuration
# Ensure webhook is added to correct channel
# Check channel permissions

# Test with different webhook
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/NEW/WEBHOOK"
python -c "
from src.slack import notify
notify('Test with new webhook')
"
```

### 6. Queue Management Issues

#### Supabase Connection

**Symptoms:**
- "Supabase client not available" warnings
- Queue not loading

**Solutions:**
```bash
# Check Supabase credentials
python -c "
import os
print('Supabase URL:', os.getenv('SUPABASE_URL'))
print('Service Key:', os.getenv('SUPABASE_SERVICE_ROLE_KEY')[:10] + '...' if os.getenv('SUPABASE_SERVICE_ROLE_KEY') else None)
"

# Test Supabase connection
python -c "
from src.main import _get_supabase
sb = _get_supabase()
if sb:
    print('Supabase: OK')
else:
    print('Supabase: Not configured')
"
```

#### CSV Queue Issues

**Symptoms:**
- Queue file not found
- Invalid CSV format

**Solutions:**
```bash
# Check queue file exists
ls -la data/creatives_queue.csv

# Validate CSV format
python -c "
import pandas as pd
try:
    df = pd.read_csv('data/creatives_queue.csv')
    print(f'CSV: OK ({len(df)} rows)')
    print('Columns:', list(df.columns))
except Exception as e:
    print(f'CSV Error: {e}')
"

# Create sample queue file if missing
cat > data/creatives_queue.csv << 'EOF'
video_id,filename,avatar,visual_style,script,status
1438715257185990,video1.mp4,avatar1.jpg,style1,script1,pending
1438715257185991,video2.mp4,avatar2.jpg,style2,script2,pending
EOF
```

### 7. Performance Issues

#### Slow Execution

**Symptoms:**
- Long execution times
- Timeout errors

**Solutions:**
```bash
# Check execution time
time python src/main.py --dry-run

# Optimize API settings
export META_TIMEOUT=60
export META_RETRY_MAX=3
export META_BACKOFF_BASE=0.5

# Reduce concurrent operations
export META_WRITE_COOLDOWN_SEC=10
```

#### Memory Issues

**Symptoms:**
- Out of memory errors
- Slow performance

**Solutions:**
```bash
# Check memory usage
python src/main.py --dry-run &
ps aux | grep python

# Optimize data processing
# Reduce queue size
# Use pagination for large datasets
```

### 8. Business Logic Issues

#### Rule Evaluation Problems

**Symptoms:**
- Ads not being killed/promoted as expected
- Unexpected behavior

**Solutions:**
```bash
# Test rule evaluation
python src/main.py --explain

# Check specific stage
python src/main.py --stage testing --explain

# Verify rule configuration
python -c "
import yaml
with open('config/rules.yaml') as f:
    rules = yaml.safe_load(f)
print('Testing rules:', rules.get('testing', {}).get('kill', []))
"
```

#### Threshold Issues

**Symptoms:**
- Thresholds not working as expected
- Performance metrics incorrect

**Solutions:**
```bash
# Check threshold configuration
python -c "
import yaml
with open('config/rules.yaml') as f:
    rules = yaml.safe_load(f)
thresholds = rules.get('thresholds', {})
print('CPA thresholds:', thresholds.get('cpa', {}))
print('ROAS thresholds:', thresholds.get('roas', {}))
"

# Test with simulation mode
python src/main.py --simulate --explain
```

## Debugging Techniques

### Enable Debug Logging

```bash
# Set debug environment variable
export DEBUG=true

# Run with verbose output
python src/main.py --dry-run --explain
```

### Log Analysis

```bash
# View recent logs
tail -f data/actions.log.jsonl

# Filter for specific issues
grep "error" data/actions.log.jsonl
grep "testing" data/actions.log.jsonl
grep "kill" data/actions.log.jsonl

# View database logs
sqlite3 data/state.sqlite "SELECT * FROM logs ORDER BY created_at DESC LIMIT 10;"
```

### Step-by-Step Debugging

```bash
# 1. Test configuration
python src/main.py --dry-run --explain

# 2. Test specific stage
python src/main.py --stage testing --dry-run --explain

# 3. Test with simulation
python src/main.py --simulate --explain

# 4. Test API access
python -c "
from src.meta_client import MetaClient, AccountAuth, ClientConfig
# Test API connection
"

# 5. Test database
python -c "
from src.storage import Store
# Test database access
"
```

## Recovery Procedures

### System Recovery

```bash
# 1. Stop all processes
pkill -f "python src/main.py"

# 2. Check system health
python -c "
from src.main import health_check
# Run health check
"

# 3. Fix configuration issues
# Edit config files as needed

# 4. Test with dry run
python src/main.py --dry-run

# 5. Resume normal operation
python src/main.py --profile production
```

### Data Recovery

```bash
# 1. Backup current state
cp data/state.sqlite data/state.sqlite.backup.$(date +%Y%m%d)

# 2. Check database integrity
sqlite3 data/state.sqlite "PRAGMA integrity_check;"

# 3. If corrupted, restore from backup
cp data/state.sqlite.backup.$(date +%Y%m%d) data/state.sqlite

# 4. Test database access
python -c "
from src.storage import Store
store = Store('data/state.sqlite')
# Test access
"
```

### Configuration Recovery

```bash
# 1. Backup configuration
cp config/settings.yaml config/settings.yaml.backup
cp config/rules.yaml config/rules.yaml.backup

# 2. Restore from backup
cp config/settings.yaml.backup config/settings.yaml
cp config/rules.yaml.backup config/rules.yaml

# 3. Validate configuration
python -c "
import yaml
# Validate YAML files
"

# 4. Test configuration
python src/main.py --dry-run
```

## Getting Help

### Self-Service Resources

1. **Check logs first**
   ```bash
   tail -f data/actions.log.jsonl
   ```

2. **Review configuration**
   ```bash
   python src/main.py --dry-run --explain
   ```

3. **Test with simulation**
   ```bash
   python src/main.py --simulate
   ```

### Support Channels

1. **Documentation**
   - Check this troubleshooting guide
   - Review main README
   - Check configuration guides

2. **Log Analysis**
   - Examine error messages
   - Check system logs
   - Review database logs

3. **Community Support**
   - Create issue in repository
   - Check existing issues
   - Review pull requests

### Escalation

If self-service doesn't resolve the issue:

1. **Gather Information**
   - System logs
   - Configuration files
   - Error messages
   - Steps to reproduce

2. **Create Issue**
   - Include all gathered information
   - Describe expected vs actual behavior
   - Provide steps to reproduce

3. **Contact Support**
   - Use appropriate support channel
   - Include all relevant information
   - Be specific about the problem

## Prevention

### Regular Maintenance

```bash
# Daily health checks
python src/main.py --dry-run

# Weekly configuration review
python src/main.py --simulate --since 2024-01-01 --until 2024-01-07

# Monthly system review
# Check logs, performance, configuration
```

### Monitoring

```bash
# Set up monitoring
# Monitor system health
# Check API usage
# Review performance metrics
```

### Backup Procedures

```bash
# Regular backups
cp data/state.sqlite data/backups/state.sqlite.$(date +%Y%m%d)
cp config/settings.yaml config/backups/settings.yaml.$(date +%Y%m%d)
cp config/rules.yaml config/backups/rules.yaml.$(date +%Y%m%d)
```

## Best Practices

### Development

1. **Always test with dry-run first**
2. **Use staging profile for testing**
3. **Test configuration changes incrementally**
4. **Keep backups of working configurations**

### Production

1. **Monitor system health regularly**
2. **Set up proper alerting**
3. **Keep configuration in version control**
4. **Document any custom changes**

### Troubleshooting

1. **Check logs first**
2. **Use explain mode for debugging**
3. **Test with simulation mode**
4. **Verify configuration step by step**
