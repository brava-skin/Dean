# Dean ML System Setup Guide

This guide will help you set up the complete self-learning Meta Ads automation system with advanced ML capabilities.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Install all ML dependencies
pip install -r requirements.txt

# Verify ML packages are installed
python -c "import xgboost, sklearn, pandas, numpy; print('✅ ML packages installed')"
```

### 2. Supabase Database Setup

#### Step 1: Create Supabase Project
1. Go to [supabase.com](https://supabase.com)
2. Create a new project
3. Note your project URL and service role key

#### Step 2: Run Database Schema
```bash
# Copy your Supabase credentials
export SUPABASE_URL="your-project-url"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"

# Run the main schema
psql -h your-db-host -U postgres -d postgres -f supabase_schema.sql

# Run security fixes
psql -h your-db-host -U postgres -d postgres -f supabase_security_fixes.sql
```

#### Step 3: Verify Database Setup
```sql
-- Check if all tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'ad_lifecycle', 'performance_metrics', 'ml_models', 'ml_predictions',
    'learning_events', 'adaptive_rules', 'system_health'
);

-- Check RLS policies
SELECT schemaname, tablename, policyname 
FROM pg_policies 
WHERE schemaname = 'public';
```

### 3. Environment Configuration

Create a `.env` file with all required variables:

```bash
# Meta Ads API
FB_APP_ID=your-app-id
FB_APP_SECRET=your-app-secret
FB_ACCESS_TOKEN=your-access-token
FB_AD_ACCOUNT_ID=your-account-id
FB_PIXEL_ID=your-pixel-id
FB_PAGE_ID=your-page-id
STORE_URL=your-store-url
IG_ACTOR_ID=your-instagram-actor-id

# Supabase ML System
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_TABLE=meta_creatives

# Optional: Timezone and other settings
TIMEZONE=Europe/Amsterdam
```

### 4. Test ML System

```bash
# Test standard mode (legacy system)
python src/main.py --dry-run

# Test ML-enhanced mode
python src/main.py --ml-mode --dry-run

# Test specific stage with ML
python src/main.py --ml-mode --stage testing --dry-run
```

## 🧠 ML System Features

### Core ML Components

1. **XGBoost Prediction Engine** (`ml_intelligence.py`)
   - CPA prediction with confidence intervals
   - ROAS forecasting
   - Purchase probability estimation
   - Cross-stage transfer learning

2. **Adaptive Rules Engine** (`adaptive_rules.py`)
   - Dynamic threshold adjustment
   - Performance-based rule adaptation
   - Confidence-weighted decisions

3. **Performance Tracking** (`performance_tracking.py`)
   - Advanced fatigue detection
   - Performance decay analysis
   - Half-life estimation

4. **ML Reporting** (`ml_reporting.py`)
   - Predictive insights
   - System health monitoring
   - Automated recommendations

### Database Schema

The system uses a comprehensive Supabase schema with:

- **Core Tables**: `ad_lifecycle`, `performance_metrics`, `ml_models`
- **ML Tables**: `ml_predictions`, `learning_events`, `adaptive_rules`
- **Analytics Tables**: `performance_patterns`, `temporal_analysis`
- **Creative Intelligence**: `creative_intelligence`, `creative_similarity`

## 🔧 Configuration

### ML System Settings

The ML system can be configured through environment variables or by modifying the ML config:

```python
# In your Python code
from ml_intelligence import MLConfig

config = MLConfig(
    retrain_frequency_hours=24,
    prediction_horizon_hours=24,
    confidence_threshold=0.7,
    learning_rate=0.1
)
```

### Adaptive Rules Configuration

```python
# In your Python code
from adaptive_rules import RuleConfig

rule_config = RuleConfig(
    target_cpa=27.50,
    target_roas=2.0,
    target_ctr=0.008,
    learning_rate=0.1,
    max_adjustment_pct=0.2
)
```

## 📊 Monitoring and Reporting

### ML Insights Dashboard

The system provides comprehensive ML insights:

- **Prediction Accuracy**: Track ML model performance
- **Learning Velocity**: Monitor system learning rate
- **Intelligence Scores**: Ad-level intelligence metrics
- **Fatigue Detection**: Performance decay alerts

### Slack Notifications

ML-enhanced notifications include:

- 🧠 **ML Intelligence**: Prediction confidence and accuracy
- 📈 **Performance Forecasts**: Future CPA/ROAS predictions
- ⚠️ **Fatigue Alerts**: Early warning system for ad fatigue
- 💡 **Recommendations**: ML-generated optimization suggestions

## 🚨 Troubleshooting

### Common Issues

1. **Supabase Connection Errors**
   ```bash
   # Verify credentials
   python -c "from supabase import create_client; print('✅ Supabase connection works')"
   ```

2. **ML Model Training Failures**
   ```bash
   # Check data availability
   python -c "from src.ml_intelligence import create_ml_system; print('✅ ML system initializes')"
   ```

3. **Permission Errors**
   ```bash
   # Run security fixes
   psql -h your-db-host -U postgres -d postgres -f supabase_security_fixes.sql
   ```

### Performance Optimization

1. **Database Indexing**
   - Ensure all foreign keys have indexes
   - Add composite indexes for common queries

2. **ML Model Optimization**
   - Adjust retrain frequency based on data volume
   - Tune hyperparameters for your specific use case

3. **Memory Management**
   - Monitor ML model memory usage
   - Implement model pruning for large datasets

## 🔄 Maintenance

### Regular Maintenance Tasks

1. **Model Retraining**
   - Models auto-retrain every 24 hours
   - Manual retraining: `python -c "from src.ml_intelligence import create_ml_system; system = create_ml_system(url, key); system.initialize_models()"`

2. **Database Cleanup**
   - Old predictions are auto-expired after 7 days
   - Performance metrics are kept for 90 days

3. **Performance Monitoring**
   - Check system health scores
   - Monitor learning velocity
   - Review prediction accuracy

### Backup and Recovery

1. **Database Backups**
   ```bash
   # Create backup
   pg_dump -h your-db-host -U postgres your-database > backup.sql
   
   # Restore backup
   psql -h your-db-host -U postgres your-database < backup.sql
   ```

2. **ML Model Backups**
   - Models are stored in Supabase `ml_models` table
   - Export models: `SELECT model_data FROM ml_models WHERE is_active = true;`

## 📈 Advanced Usage

### Custom ML Models

You can extend the system with custom models:

```python
# Add custom model type
from ml_intelligence import MLIntelligenceSystem

class CustomMLSystem(MLIntelligenceSystem):
    def train_custom_model(self, model_type, stage, target_col):
        # Your custom training logic
        pass
```

### Custom Rules

Add custom adaptive rules:

```python
# Custom rule adaptation
from adaptive_rules import IntelligentRuleEngine

def custom_rule_adaptation(self, rule, performance_data, ml_predictions):
    # Your custom rule logic
    pass
```

## 🎯 Best Practices

1. **Data Quality**
   - Ensure consistent data collection
   - Monitor for missing or invalid data
   - Regular data quality checks

2. **Model Performance**
   - Track prediction accuracy over time
   - Adjust confidence thresholds based on performance
   - Monitor for model drift

3. **System Monitoring**
   - Set up alerts for system health
   - Monitor learning velocity
   - Track ML recommendation effectiveness

4. **Security**
   - Regularly update Supabase security policies
   - Monitor access logs
   - Keep API keys secure

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review Supabase logs for database issues
3. Check ML model performance in the dashboard
4. Verify all environment variables are set correctly

The ML system is designed to be self-learning and adaptive, but proper setup and monitoring are essential for optimal performance.
