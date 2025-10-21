# Setup Guide for Real Data Testing

## 🚀 How to Test the ML System with Real Meta Ads Data

### Step 1: Create Environment File

Create a `.env` file in the project root with your credentials:

```bash
# Meta Ads API Credentials
FB_APP_ID=your-app-id
FB_APP_SECRET=your-app-secret
FB_ACCESS_TOKEN=your-access-token
FB_AD_ACCOUNT_ID=your-account-id
FB_PIXEL_ID=your-pixel-id
FB_PAGE_ID=your-page-id
STORE_URL=your-store-url
IG_ACTOR_ID=your-instagram-actor-id

# Supabase Credentials (for ML mode)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_TABLE=meta_creatives

# Optional settings
TIMEZONE=Europe/Amsterdam
MODE=production
DRY_RUN=false
```

### Step 2: Get Meta API Credentials

1. **Go to Meta for Developers**: https://developers.facebook.com/
2. **Create/Select App**: Choose your Facebook app
3. **Get App ID & Secret**: From App Settings
4. **Generate Access Token**: 
   - Go to Graph API Explorer
   - Select your app
   - Generate token with `ads_read`, `ads_management` permissions
5. **Get Account ID**: From Meta Ads Manager
6. **Get Pixel ID**: From Events Manager
7. **Get Page ID**: From your Facebook page
8. **Store URL**: Your website URL
9. **Instagram Actor ID**: From Instagram Business account

### Step 3: Get Supabase Credentials

1. **Go to Supabase**: https://supabase.com/
2. **Create/Select Project**: Choose your project
3. **Get URL**: From Project Settings > API
4. **Get Service Role Key**: From Project Settings > API > service_role key
5. **Deploy Schema**: Run the database schema files

### Step 4: Test the System

#### Test 1: Standard Mode (Legacy System)
```bash
# This should work exactly like before
python3 src/main.py --dry-run
```

#### Test 2: ML-Enhanced Mode (New System)
```bash
# This should show ML enhancements
python3 src/main.py --ml-mode --dry-run
```

#### Test 3: Specific Stage Testing
```bash
# Test ML system on specific stage
python3 src/main.py --ml-mode --stage testing --dry-run
```

### Step 5: Validate ML System

#### What You Should See:

**Standard Mode:**
- Same ad loading behavior
- Same rule application
- Same output format
- No breaking changes

**ML Mode:**
- ML intelligence analysis
- Adaptive rules adjustments
- Predictive insights
- Cross-stage learning events
- Performance tracking data

#### Expected ML Output:
```
🧠 ML Intelligence Analysis:
   - Data depth: 30 days
   - Confidence: 0.85
   - Learning velocity: 0.12
   - Pattern recognition: 3 patterns found

⚙️ Adaptive Rules Engine:
   - CPA threshold: €27.50 → €26.80 (ML adjusted)
   - CTR minimum: 0.008 → 0.009 (trend-based)
   - ROAS target: 2.0 → 2.1 (performance-based)

📈 Predictive Insights:
   - Ad A: 78% chance of promotion (confidence: 0.82)
   - Ad B: 45% fatigue risk (confidence: 0.76)
   - Ad C: €24.50 predicted CPA (confidence: 0.88)
```

### Step 6: Gradual Migration

1. **Start with Test Account**: Use ML mode on a test Meta account
2. **Monitor Performance**: Track ML system effectiveness
3. **Compare Results**: Standard vs ML mode performance
4. **Fine-tune**: Adjust based on real-world results
5. **Full Migration**: Replace old system completely

### Troubleshooting

#### Common Issues:

1. **Meta API Errors**
   - Check credentials are correct
   - Verify account has proper permissions
   - Ensure API version compatibility

2. **Supabase Connection Issues**
   - Verify SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
   - Check database schema is deployed
   - Ensure RLS policies are correct

3. **ML Model Training Issues**
   - Check if enough data is available
   - Verify feature extraction works
   - Monitor model performance metrics

### Success Criteria

- ✅ **Backward Compatibility**: Old system behavior preserved
- ✅ **ML Enhancements**: New intelligence features working
- ✅ **Data Integrity**: No data loss or corruption
- ✅ **Performance**: System runs efficiently
- ✅ **Accuracy**: ML predictions are reasonable
- ✅ **Learning**: System improves over time

### Next Steps After Testing

1. **Gradual Rollout**: Start with ML mode on test accounts
2. **Monitor Performance**: Track ML system effectiveness
3. **Fine-tune Models**: Adjust based on real-world results
4. **Full Migration**: Replace old system completely
5. **Continuous Learning**: Let system improve over time

The ML system is designed to be a drop-in replacement for the old system while adding advanced intelligence capabilities.
