# 🚀 Dean GitHub Actions Setup Guide

## ✅ **Migration from DigitalOcean to GitHub Actions**

### **Why GitHub Actions?**
- ✅ **Simpler**: No server management
- ✅ **Reliable**: GitHub's infrastructure
- ✅ **Cost-effective**: Free for public repos
- ✅ **Easy monitoring**: Built-in logs and status
- ✅ **30-minute schedule**: Perfect for ad automation

## 🔧 **Setup Instructions**

### **1. Repository Secrets**
Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

#### **Meta Ads API Configuration:**
```
FB_APP_ID=your_app_id
FB_APP_SECRET=your_app_secret
FB_ACCESS_TOKEN=your_access_token
FB_AD_ACCOUNT_ID=your_ad_account_id
FB_PIXEL_ID=your_pixel_id
FB_PAGE_ID=your_page_id
IG_ACTOR_ID=your_ig_actor_id
```

#### **Supabase Configuration:**
```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives
```

#### **Business Configuration:**
```
STORE_URL=your_store_url
SLACK_WEBHOOK_URL=your_slack_webhook
BREAKEVEN_CPA=27.5
COGS_PER_PURCHASE=your_cogs
USD_EUR_RATE=your_exchange_rate
```

### **2. Workflow Features**

#### **Schedule:**
- **Every 30 minutes**: `*/30 * * * *`
- **Manual runs**: Available via GitHub Actions tab

#### **4-Day Testing System:**
- ✅ Maximum 4 days per creative
- ✅ Mandatory kill after 4 days
- ✅ 4 creatives always running
- ✅ €50 budget distribution

#### **Metric-Based Rules:**
- ✅ CTR <0.8% after €20 = Kill
- ✅ ATC rate <0.5% after €40 = Kill
- ✅ CPM >€120 after €15 = Kill
- ✅ Cost per ATC >€15 = Kill
- ✅ Impression drop 30%+ for 2 days = Kill

#### **Rate Limiting:**
- ✅ 2-second request delay
- ✅ Single concurrent request
- ✅ 5 retry attempts
- ✅ Exponential backoff

### **3. Monitoring**

#### **Check Status:**
1. Go to your GitHub repository
2. Click "Actions" tab
3. View "Dean 4-Day Testing System" workflow
4. Check latest run status

#### **View Logs:**
1. Click on latest workflow run
2. Click on "dean-automation" job
3. View detailed logs

#### **Manual Run:**
1. Go to Actions tab
2. Select "Dean 4-Day Testing System"
3. Click "Run workflow"
4. Click "Run workflow" button

### **4. Disable DigitalOcean**

#### **Stop DigitalOcean Service:**
```bash
# SSH into your DigitalOcean droplet
ssh root@your-droplet-ip

# Stop the service
sudo systemctl stop dean-automation.service
sudo systemctl disable dean-automation.service

# Optional: Shutdown the droplet
sudo shutdown -h now
```

## 🎯 **Benefits of GitHub Actions**

### **vs DigitalOcean:**
- ✅ **No server management**
- ✅ **No resource limits**
- ✅ **Built-in monitoring**
- ✅ **Easy scaling**
- ✅ **Free for public repos**

### **4-Day Testing System:**
- ✅ **Aggressive testing**: 4-day maximum
- ✅ **Metric-based kills**: CTR, ATC, CPM rules
- ✅ **4-creative system**: Always maintain 4 creatives
- ✅ **Budget efficiency**: €12.5 per creative

## 📊 **Expected Performance**

### **Schedule:**
- **Frequency**: Every 30 minutes
- **Duration**: ~2-3 minutes per run
- **Reliability**: 99.9% uptime

### **Ad Management:**
- **Testing Phase**: 4 days maximum
- **Creative Rotation**: Automatic every 4 days
- **Performance Rules**: Metric-based kills
- **Budget Distribution**: €50 across 4 creatives

## 🚀 **Next Steps**

1. **Add repository secrets** (see above)
2. **Push to GitHub** (workflow file is ready)
3. **Test manual run** (Actions → Run workflow)
4. **Monitor first few runs**
5. **Disable DigitalOcean** (optional)

Your Dean system will now run every 30 minutes on GitHub Actions with the new 4-day testing system! 🎉
