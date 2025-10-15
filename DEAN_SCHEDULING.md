# 🕐 Dean Automation Scheduling

Your Dean project is a **Meta advertising automation system** that needs to run regularly to manage your ad campaigns.

## 🎯 What Dean Does

- **Testing Stage**: Tests new creative assets with controlled budgets
- **Validation Stage**: Validates promising creatives with extended testing  
- **Scaling Stage**: Scales winning creatives with intelligent budget allocation
- **Monitoring**: Sends Slack notifications and maintains account health

## 🚀 How to Run Dean

### Option 1: Manual Run (One-time)
```bash
# Run all stages
run-dean-automation.bat

# Run specific stage
run-dean-stages.bat
```

### Option 2: Continuous Scheduling (Recommended)
```bash
# Run every 2 hours automatically
schedule-dean.bat
```

### Option 3: Windows Task Scheduler (Production)

1. **Open Task Scheduler** (search "Task Scheduler" in Start menu)
2. **Create Basic Task**:
   - Name: "Dean Automation"
   - Description: "Meta advertising automation"
3. **Set Trigger**: "Daily" or "At startup"
4. **Set Action**: "Start a program"
   - Program: `C:\Users\Jede Sno\Documents\Dean\run-dean-automation.bat`
5. **Advanced Settings**:
   - ✅ Run whether user is logged on or not
   - ✅ Run with highest privileges
   - ✅ Repeat task every 2 hours for 24 hours

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the `dean/` directory with:

```bash
# Meta API Credentials
FB_APP_ID=your_app_id
FB_APP_SECRET=your_app_secret
FB_ACCESS_TOKEN=your_access_token
FB_AD_ACCOUNT_ID=your_account_id
FB_PIXEL_ID=your_pixel_id
FB_PAGE_ID=your_page_id

# Optional: Instagram
IG_ACTOR_ID=your_instagram_actor_id

# Store Configuration
STORE_URL=https://your-store.com

# Slack Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Optional: Supabase (for queue management)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives

# Optional: Economics
BREAKEVEN_CPA=34
COGS_PER_PURCHASE=15
USD_EUR_RATE=0.92
```

### Configuration Files
- `dean/config/settings.yaml`: Main configuration
- `dean/config/rules.yaml`: Business rules and thresholds

## 🔄 Automation Flow

1. **Testing Stage**:
   - Launches new creatives from queue
   - Monitors performance with budget controls
   - Kills underperforming ads
   - Promotes winners to validation

2. **Validation Stage**:
   - Extended testing with higher budgets
   - Stricter performance requirements
   - Multi-day stability checks
   - Promotes winners to scaling

3. **Scaling Stage**:
   - Intelligent budget scaling
   - Portfolio management
   - Creative duplication for winners
   - Advanced pacing controls

## 📊 Monitoring

- **Slack Notifications**: Real-time updates on campaign performance
- **Logging**: Detailed logs in `dean/data/` directory
- **Health Checks**: Account-level monitoring and guardrails

## 🛠️ Troubleshooting

### Common Issues
1. **Missing Environment Variables**: Check `.env` file
2. **API Rate Limits**: Increase cooldown settings
3. **Database Lock**: Check for concurrent runs
4. **Slack Failures**: Verify webhook URL

### Debug Mode
```bash
# Run in simulation mode (no actual changes)
python src/main.py --simulate

# Run with detailed logging
python src/main.py --explain
```

## 🎯 Best Practices

1. **Start with Simulation**: Test with `--simulate` first
2. **Monitor Slack**: Watch for notifications and errors
3. **Check Logs**: Review `dean/data/` for detailed information
4. **Gradual Scaling**: Start with conservative budgets
5. **Regular Monitoring**: Check account performance daily

---

**🎉 Your Dean automation is now ready to run! Use the batch files to start your Meta advertising automation.**
