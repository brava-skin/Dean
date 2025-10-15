# Usage Guide

This guide covers how to use Dean automation system effectively, including command-line options, execution modes, and best practices.

## Command Line Interface

### Basic Usage

```bash
# Run all stages (testing, validation, scaling)
python src/main.py

# Run specific stage only
python src/main.py --stage testing
python src/main.py --stage validation
python src/main.py --stage scaling

# Dry run mode (no actual changes)
python src/main.py --dry-run

# Simulation mode (explain decisions without acting)
python src/main.py --simulate
```

### Command Line Options

#### Core Options

```bash
# Stage Selection
--stage {all,testing,validation,scaling}
    Select which automation stage to run
    Default: all

# Configuration Files
--settings PATH
    Path to settings YAML file
    Default: config/settings.yaml

--rules PATH
    Path to rules YAML file
    Default: config/rules.yaml

--schema PATH
    Path to JSON schema for validation
    Default: config/schema.settings.yaml
```

#### Execution Modes

```bash
# Dry Run Mode
--dry-run
    Execute without making actual changes to Meta
    Useful for testing and validation

# Simulation Mode
--simulate
    Shadow mode: log intended actions only
    No actual API calls made

# Explain Mode
--explain
    Print decisions without acting
    Shows what would be done without executing
```

#### Profile Selection

```bash
# Profile Selection
--profile {production,staging}
    Select execution profile
    Default: from settings or environment

# Staging Profile
python src/main.py --profile staging
    Uses staging configuration with relaxed thresholds

# Production Profile
python src/main.py --profile production
    Uses production configuration with full enforcement
```

#### Simulation Options

```bash
# Date Range Simulation
--since YYYY-MM-DD
    Start date for simulation
    Example: --since 2024-01-01

--until YYYY-MM-DD
    End date for simulation
    Example: --until 2024-01-31

# Simulation Example
python src/main.py --simulate --since 2024-01-01 --until 2024-01-31
```

#### Output Control

```bash
# Disable Digest
--no-digest
    Skip daily digest generation
    Useful for testing or debugging
```

## Execution Modes

### Production Mode

```bash
# Full production execution
python src/main.py --profile production

# With specific stage
python src/main.py --stage testing --profile production
```

**Characteristics:**
- Makes actual changes to Meta campaigns
- Enforces all business rules and thresholds
- Sends notifications to Slack
- Records all actions in database
- Generates daily digests

### Staging Mode

```bash
# Staging execution
python src/main.py --profile staging

# Staging with dry run
python src/main.py --profile staging --dry-run
```

**Characteristics:**
- Relaxed thresholds for testing
- May disable certain safety checks
- Still makes API calls (unless --dry-run)
- Useful for testing configuration changes

### Dry Run Mode

```bash
# Dry run execution
python src/main.py --dry-run

# Dry run with specific stage
python src/main.py --stage testing --dry-run
```

**Characteristics:**
- No actual changes to Meta campaigns
- All API calls are simulated
- Useful for testing configuration
- Safe for development and testing

### Simulation Mode

```bash
# Simulation execution
python src/main.py --simulate

# Simulation with date range
python src/main.py --simulate --since 2024-01-01 --until 2024-01-31
```

**Characteristics:**
- No API calls made
- Uses historical data for decisions
- Shows what would have been done
- Useful for backtesting and analysis

### Explain Mode

```bash
# Explain mode
python src/main.py --explain

# Explain specific stage
python src/main.py --stage testing --explain
```

**Characteristics:**
- Shows decision logic without executing
- Useful for debugging rules
- No API calls or changes made
- Detailed output of decision process

## Stage-Specific Usage

### Testing Stage

```bash
# Run testing stage only
python src/main.py --stage testing

# Testing with dry run
python src/main.py --stage testing --dry-run

# Testing with explain mode
python src/main.py --stage testing --explain
```

**What it does:**
- Launches new ads from creative queue
- Monitors performance with budget controls
- Kills underperforming ads based on rules
- Promotes successful ads to validation
- Manages queue rotation and fairness

### Validation Stage

```bash
# Run validation stage only
python src/main.py --stage validation

# Validation with dry run
python src/main.py --stage validation --dry-run
```

**What it does:**
- Tests promoted ads with higher budgets
- Applies stricter performance requirements
- Promotes validated ads to scaling
- Manages extended testing periods
- Handles soft pass scenarios

### Scaling Stage

```bash
# Run scaling stage only
python src/main.py --stage scaling

# Scaling with dry run
python src/main.py --stage scaling --dry-run
```

**What it does:**
- Scales budgets for winning ads
- Manages portfolio allocation
- Handles creative duplication
- Implements reinvestment strategies
- Monitors scaling performance

## Advanced Usage Patterns

### Development Workflow

```bash
# 1. Test configuration
python src/main.py --dry-run --explain

# 2. Test specific stage
python src/main.py --stage testing --dry-run

# 3. Test with staging profile
python src/main.py --profile staging --dry-run

# 4. Run in production
python src/main.py --profile production
```

### Testing Configuration Changes

```bash
# Test new settings file
python src/main.py --settings config/new_settings.yaml --dry-run

# Test new rules file
python src/main.py --rules config/new_rules.yaml --dry-run

# Test both
python src/main.py --settings config/new_settings.yaml --rules config/new_rules.yaml --dry-run
```

### Backtesting and Analysis

```bash
# Simulate past performance
python src/main.py --simulate --since 2024-01-01 --until 2024-01-31

# Analyze specific date range
python src/main.py --simulate --since 2024-01-15 --until 2024-01-20 --explain
```

### Debugging Issues

```bash
# Debug with explain mode
python src/main.py --explain

# Debug specific stage
python src/main.py --stage testing --explain

# Debug with verbose output
python src/main.py --stage testing --explain --no-digest
```

## Scheduling and Automation

### Cron Scheduling

```bash
# Run every 2 hours
0 */2 * * * cd /path/to/dean && python src/main.py --profile production

# Run testing every hour
0 * * * * cd /path/to/dean && python src/main.py --stage testing --profile production

# Run validation every 4 hours
0 */4 * * * cd /path/to/dean && python src/main.py --stage validation --profile production

# Run scaling every 6 hours
0 */6 * * * cd /path/to/dean && python src/main.py --stage scaling --profile production
```

### Systemd Service

Create `/etc/systemd/system/dean-automation.service`:

```ini
[Unit]
Description=Dean Advertising Automation
After=network.target

[Service]
Type=simple
User=dean
WorkingDirectory=/path/to/dean
ExecStart=/path/to/dean/venv/bin/python src/main.py --profile production
Restart=always
RestartSec=300

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable dean-automation
sudo systemctl start dean-automation
sudo systemctl status dean-automation
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/main.py", "--profile", "production"]
```

Build and run:

```bash
docker build -t dean-automation .
docker run -d --name dean --env-file .env dean-automation
```

## Monitoring and Logging

### Log Files

```bash
# View recent logs
tail -f data/actions.log.jsonl

# View specific stage logs
grep "testing" data/actions.log.jsonl

# View error logs
grep "error" data/actions.log.jsonl
```

### Database Monitoring

```bash
# Check database status
sqlite3 data/state.sqlite "SELECT COUNT(*) FROM logs;"

# View recent activity
sqlite3 data/state.sqlite "SELECT * FROM logs ORDER BY created_at DESC LIMIT 10;"
```

### Health Checks

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

## Best Practices

### Development

1. **Always test with dry-run first**
   ```bash
   python src/main.py --dry-run
   ```

2. **Use staging profile for testing**
   ```bash
   python src/main.py --profile staging --dry-run
   ```

3. **Test configuration changes incrementally**
   ```bash
   python src/main.py --settings config/test_settings.yaml --dry-run
   ```

### Production

1. **Start with conservative settings**
2. **Monitor performance closely**
3. **Use appropriate scheduling**
4. **Set up proper monitoring**
5. **Keep backups of configuration**

### Troubleshooting

1. **Check logs first**
   ```bash
   tail -f data/actions.log.jsonl
   ```

2. **Use explain mode for debugging**
   ```bash
   python src/main.py --explain
   ```

3. **Test with simulation mode**
   ```bash
   python src/main.py --simulate
   ```

4. **Verify configuration**
   ```bash
   python src/main.py --dry-run --explain
   ```

## Common Use Cases

### Daily Operations

```bash
# Morning check
python src/main.py --stage testing --dry-run

# Full daily run
python src/main.py --profile production

# Evening validation
python src/main.py --stage validation --profile production
```

### Weekly Maintenance

```bash
# Full system check
python src/main.py --dry-run --explain

# Performance analysis
python src/main.py --simulate --since 2024-01-01 --until 2024-01-07

# Configuration review
python src/main.py --profile staging --dry-run
```

### Emergency Procedures

```bash
# Emergency stop (dry run)
python src/main.py --dry-run

# Check system status
python src/main.py --explain

# Resume operations
python src/main.py --profile production
```

## Performance Optimization

### Execution Frequency

- **Testing**: Every 1-2 hours
- **Validation**: Every 4-6 hours  
- **Scaling**: Every 6-12 hours
- **Full Run**: Daily

### Resource Management

```bash
# Limit concurrent operations
export META_WRITE_COOLDOWN_SEC=10

# Adjust retry settings
export META_RETRY_MAX=3
export META_BACKOFF_BASE=0.5
```

### Monitoring Performance

```bash
# Check execution time
time python src/main.py --stage testing

# Monitor memory usage
python src/main.py --stage testing &
ps aux | grep python
```

## Troubleshooting Usage Issues

### Common Problems

**1. Permission Errors**
```bash
# Check file permissions
ls -la data/
chmod 755 data/
```

**2. Configuration Errors**
```bash
# Validate configuration
python src/main.py --dry-run --explain
```

**3. API Rate Limits**
```bash
# Increase cooldown
export META_WRITE_COOLDOWN_SEC=30
```

**4. Database Issues**
```bash
# Check database
sqlite3 data/state.sqlite ".tables"
```

### Getting Help

1. Check the troubleshooting section in the main README
2. Review log files for error messages
3. Test with dry-run mode first
4. Use explain mode to understand decisions
5. Verify all configuration is correct
