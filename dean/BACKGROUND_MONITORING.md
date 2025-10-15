# Enhanced Monitoring System

This system enhances your existing GitHub Actions hourly ticks with intelligent alerts, periodic summaries, and humanized messaging.

## Features

### 🔄 Enhanced Hourly Ticks
- **Regular Automation**: Your existing testing, validation, and scaling stages
- **3-Hour Summaries**: Automatic metrics overview every 3 hours (3 AM, 6 AM, 9 AM, etc.)
- **Daily Summaries**: Morning reports at 8 AM of previous day's performance
- **Smart Timing**: Summaries only run during the first 5 minutes of their scheduled hour

### 🚨 Intelligent Alerts
- **Queue Empty**: Alerts when no more creatives are available for testing
- **Ad Performance**: Kills, promotions, scaling, and fatigue alerts
- **System Health**: API connectivity and database issues
- **Budget Changes**: Significant budget adjustments

### 💬 Humanized Messaging
All alerts are written like a media buyer colleague texting you:
- "Hey! Had to kill [Ad Name] - not hitting our targets"
- "Great news! [Ad Name] is performing well - moving to validation"
- "URGENT: No more creatives in the queue!"

## Usage

### Automatic with GitHub Actions (Recommended)

Your existing GitHub Actions workflow will automatically:
- Run hourly ticks as usual
- Add 3-hour summaries at 3 AM, 6 AM, 9 AM, 12 PM, 3 PM, 6 PM, 9 PM, 12 AM
- Add daily summaries at 8 AM
- Send humanized alerts for critical events

**No changes needed** - it just works with your existing setup!

### Manual Testing

```bash
# Test a single run with summaries
python src/main.py --profile production

# Test background mode (for VPS deployment)
python start_background.py
```

## Configuration

### Environment Variables

Make sure these are set in your `.env` file:

```bash
# Meta API credentials
FB_APP_ID=your_app_id
FB_APP_SECRET=your_app_secret
FB_ACCESS_TOKEN=your_access_token
FB_AD_ACCOUNT_ID=your_account_id
FB_PIXEL_ID=your_pixel_id
FB_PAGE_ID=your_page_id
IG_ACTOR_ID=your_instagram_actor_id

# Slack webhooks (all messages go to one channel by default)
SLACK_WEBHOOK_URL=your_main_webhook
# Optional: Route different message types to different channels
# SLACK_WEBHOOK_ALERTS=your_alerts_webhook
# SLACK_WEBHOOK_DIGEST=your_digest_webhook  
# SLACK_WEBHOOK_SCALE=your_scale_webhook

# Supabase (optional - for queue management)
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives
```

### Settings Configuration

The system uses your existing `config/settings.yaml` and `config/rules.yaml` files.

## Alert Types

### 🚨 Critical Alerts (Immediate)
- **Queue Empty**: No more creatives to test
- **System Errors**: API failures, database issues
- **Ad Kills**: Poor performing ads being paused

### 📊 Performance Alerts
- **Promotions**: Ads moving between stages
- **Scaling**: Budget increases for winning ads
- **Fatigue**: Ads showing performance decline

### 📈 Summary Reports
- **3-Hour Summaries**: Account metrics and top performers
- **Daily Reports**: Previous day's performance by stage

## Message Examples

### Kill Alert
```
🚨 Hey! Had to kill Keith - DGS Effect - ProductHighlights in TEST
CPA too high vs breakeven
CTR was 1.2%, ROAS 0.8 - not hitting our targets
```

### Promotion Alert
```
🎉 Great news! Keith - DGS Effect - ProductHighlights is performing well
Moving from TEST to VALID stage
Setting budget to €40/day on FB + IG
Keeping the same ad ID for tracking
```

### Queue Empty Alert
```
🚨 URGENT: No more creatives in the queue!
Need to upload new videos ASAP or we'll run out of tests
```

### Daily Summary
```
📊 Daily Report • 2024-01-15
Hey! Here's what happened yesterday (2024-01-15):

*Testing:* 3 ads, €120.50 spend, 2 purchases
*Validation:* 1 ads, €40.00 spend, 1 purchases  
*Scaling:* 2 ads, €200.00 spend, 8 purchases
```

## Monitoring Dashboard

The system provides several ways to monitor status:

### Console Output
- Startup/shutdown messages
- Error logs
- Health check results

### Slack Integration
- Real-time alerts for critical events
- Periodic summaries
- Humanized messaging

### Database Logging
- All actions logged to SQLite database
- Performance metrics tracked
- Historical data for analysis

## Troubleshooting

### Common Issues

1. **Scheduler not starting**
   - Check environment variables are set
   - Verify Meta API credentials
   - Check Slack webhook URLs

2. **No alerts being sent**
   - Verify Slack webhook configuration
   - Check network connectivity
   - Review alert cooldown settings

3. **High API usage**
   - Adjust scheduling intervals in code
   - Implement rate limiting
   - Use dry-run mode for testing

### Debug Mode

Run with verbose logging:
```bash
python src/main.py --background --dry-run
```

## Customization

### Alert Cooldowns
Modify cooldown periods in `scheduler.py`:
```python
# Alert cooldowns (hours)
QUEUE_EMPTY_COOLDOWN = 6
API_ERROR_COOLDOWN = 2
DB_ERROR_COOLDOWN = 1
```

### Message Templates
Customize alert messages in `slack.py`:
```python
def template_kill(stage: str, entity_name: str, reason: str, metrics: Dict[str, Any]):
    # Customize your kill message template
    text = f"🚨 Hey! Had to kill {clean_name} in {stage}\n{reason}"
```

### Scheduling
Modify intervals in `scheduler.py`:
```python
# Change scheduling intervals
schedule.every().hour.at(":00").do(self._run_hourly_tick)
schedule.every(3).hours.do(self._run_3h_summary)
schedule.every().day.at("08:00").do(self._run_daily_summary)
```

## Production Deployment

### Systemd Service (Linux)
Create `/etc/systemd/system/dean-monitor.service`:
```ini
[Unit]
Description=Dean Background Monitor
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/dean
ExecStart=/usr/bin/python3 /path/to/dean/start_background.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable dean-monitor
sudo systemctl start dean-monitor
```

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "start_background.py"]
```

### Process Management
Use PM2 for Node.js-style process management:
```bash
npm install -g pm2
pm2 start start_background.py --name dean-monitor
pm2 save
pm2 startup
```

## Security Considerations

- Store credentials in environment variables
- Use service accounts with minimal permissions
- Implement webhook authentication
- Regular security updates
- Monitor for unusual activity

## Performance Optimization

- Use connection pooling for database
- Implement caching for frequent API calls
- Batch operations where possible
- Monitor memory usage
- Set appropriate timeouts

## Support

For issues or questions:
1. Check the console output for errors
2. Review the database logs
3. Test individual components
4. Check Meta API status
5. Verify Slack webhook configuration
