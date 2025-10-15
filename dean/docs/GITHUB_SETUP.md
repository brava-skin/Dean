# 🔧 GitHub Actions Setup for Dean Automation

Your Dean automation will now run on GitHub Actions every hour, while your auto-sync keeps your code updated.

## 🚀 What's Set Up

- ✅ **Auto-sync**: Keeps your code synced to GitHub every 30 seconds
- ✅ **Dean Automation**: Runs on GitHub Actions every hour
- ✅ **Manual triggers**: You can run automation manually from GitHub
- ✅ **Logs**: Automation logs are saved as artifacts

## 🔐 Required GitHub Secrets

Go to your repository: https://github.com/brava-skin/Dean/settings/secrets/actions

### **Meta API Credentials** (Required)
```
FB_APP_ID=your_facebook_app_id
FB_APP_SECRET=your_facebook_app_secret
FB_ACCESS_TOKEN=your_facebook_access_token
FB_AD_ACCOUNT_ID=your_ad_account_id
FB_PIXEL_ID=your_pixel_id
FB_PAGE_ID=your_page_id
```

### **Optional: Instagram**
```
IG_ACTOR_ID=your_instagram_actor_id
```

### **Store Configuration** (Required)
```
STORE_URL=https://your-store.com
```

### **Slack Notifications** (Recommended)
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_WEBHOOK_URL_ERRORS=https://hooks.slack.com/services/YOUR/ERROR/WEBHOOK
```

### **Optional: Supabase** (For queue management)
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives
```

### **Optional: Economics**
```
BREAKEVEN_CPA=34
COGS_PER_PURCHASE=15
USD_EUR_RATE=0.92
```

### **Optional: Timezone**
```
TIMEZONE=Europe/Amsterdam
```

## 🕐 How It Works

### **Automatic Schedule**
- **Every hour**: Dean automation runs automatically
- **All stages**: Testing → Validation → Scaling
- **Production mode**: Real changes to your Meta ads
- **Logs saved**: Available in GitHub Actions artifacts

### **Manual Triggers**
1. Go to **Actions** tab in your GitHub repository
2. Click **Dean Automation**
3. Click **Run workflow**
4. Choose which stage to run (all, testing, validation, scaling)

### **Monitoring**
- **GitHub Actions**: Check status and logs
- **Slack notifications**: Real-time updates (if configured)
- **Artifacts**: Download logs from each run

## 🔄 Auto-Sync + GitHub Actions

### **Auto-Sync (Local)**
- **Purpose**: Sync your code changes to GitHub
- **Frequency**: Every 30 seconds
- **Scope**: Code files only
- **Status**: Running on your computer

### **Dean Automation (GitHub)**
- **Purpose**: Run Meta advertising automation
- **Frequency**: Every hour
- **Scope**: Meta API calls, ad management
- **Status**: Running on GitHub servers

## 🛠️ Setup Steps

### 1. **Add GitHub Secrets**
1. Go to: https://github.com/brava-skin/Dean/settings/secrets/actions
2. Click **New repository secret**
3. Add each secret with the exact name and value

### 2. **Test the Workflow**
1. Go to **Actions** tab
2. Click **Dean Automation**
3. Click **Run workflow**
4. Choose **all** stages
5. Click **Run workflow**

### 3. **Monitor Results**
1. Click on the workflow run
2. Check the logs for any errors
3. Download artifacts if needed

## 📊 What Happens Every Hour

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

## 🚨 Troubleshooting

### **Common Issues**
1. **Missing secrets**: Check all required secrets are set
2. **API errors**: Verify Meta API credentials
3. **Slack failures**: Check webhook URLs
4. **Timeout**: Automation has 30-minute timeout

### **Debug Mode**
- Use manual triggers to test specific stages
- Check logs in GitHub Actions
- Download artifacts for detailed logs

## 🎯 Benefits

- ✅ **Runs 24/7**: No need to keep computer on
- ✅ **Reliable**: GitHub handles scheduling
- ✅ **Scalable**: Can handle multiple accounts
- ✅ **Monitored**: Full logs and status tracking
- ✅ **Flexible**: Manual triggers for testing

---

**🎉 Your Dean automation now runs on GitHub Actions every hour, while auto-sync keeps your code updated!**
