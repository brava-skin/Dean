# Dean - Self-Learning Meta Ads Automation System

> **Next-generation ML-enhanced advertising automation for Meta (Facebook) advertising platforms**

Dean is a **fully self-learning Meta Ads automation** built around one account. It continuously learns from campaign data across **Testing → Validation → Scaling**, identifies the signals that predict purchases, and dynamically adjusts all rules to **keep CPA consistently below €27.50** while scaling safely and profitably.

## 🧠 ML Intelligence Features

- **XGBoost Prediction Engines**: Advanced ML models for performance forecasting
- **Cross-Stage Transfer Learning**: Knowledge sharing between all stages
- **Adaptive Rules Engine**: Dynamic threshold adjustment based on ML insights
- **Performance Decay Tracking**: Advanced fatigue detection and prevention
- **Temporal Modeling**: Time-series analysis and trend prediction
- **Creative Intelligence**: Similarity analysis and performance pattern recognition
- **Predictive Reporting**: ML-enhanced insights and recommendations

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd dean

# Quick setup (macOS)
chmod +x scripts/setup_macos.sh
./scripts/setup_macos.sh

# Or manual setup
pip install -r requirements.txt
cp .env.example .env  # Configure your environment
```

## 🎯 What Dean Does

Dean automates your Meta advertising with a **3-stage pipeline**:

```
Creative Queue → Testing → Validation → Scaling
     ↓             ↓         ↓          ↓
   Supabase/CSV   Budget    Extended   Portfolio
   File Input     Control   Testing    Management
```

### 🔥 Key Features

- **🧪 Advanced Testing**: 7-tier performance system with learning acceleration
- **✅ Smart Validation**: Extended testing with performance thresholds  
- **📈 Intelligent Scaling**: Portfolio management and budget optimization
- **🛡️ Account Health**: Comprehensive monitoring and safety guardrails
- **📊 Real-time Alerts**: Slack notifications with European formatting
- **🔄 Dynamic Thresholds**: Auto-detects Meta's billing thresholds

## 🚀 Usage

### ML-Enhanced Mode (Default)
```bash
# Run with ML intelligence (default mode)
python src/main.py

# Run specific stage with ML
python src/main.py --stage testing

# Background ML learning mode
python src/main.py --background

# Use production startup script
./start_ml_system.sh
```

### Legacy Mode (Fallback)
```bash
# Disable ML and use legacy system
python src/main.py --no-ml

# Legacy system for specific stage
python src/main.py --no-ml --stage testing
```

### Production Deployment
```bash
# GitHub Actions will automatically deploy
# Set up secrets in GitHub repository:
# - Meta API credentials (FB_APP_ID, FB_ACCESS_TOKEN, etc.)
# - Supabase credentials (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
# - Slack webhook (SLACK_WEBHOOK_URL)
```

### ML System Requirements
- **Supabase**: Database for ML data storage
- **Environment Variables**: `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
- **Dependencies**: All ML packages from `requirements.txt`
- **Fallback**: Automatically falls back to legacy system if ML unavailable

### Additional Options
```bash
# Dry run (no changes)
python src/main.py --dry-run

# Simulation mode
python src/main.py --simulate
```

## ⚙️ Configuration

### Environment Setup

Create a `.env` file with your Meta API credentials:

```bash
# Meta API Credentials
FB_APP_ID=your_app_id
FB_APP_SECRET=your_app_secret
FB_ACCESS_TOKEN=your_access_token
FB_AD_ACCOUNT_ID=your_account_id

# Slack Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Optional: Supabase (for queue management)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### Configuration Files

- **`config/settings.yaml`**: Main configuration (budgets, IDs, thresholds)
- **`config/rules.yaml`**: Business rules and performance thresholds

## 📊 Recent Updates

### 🆕 Dynamic Billing Threshold Detection
- Automatically detects Meta's current auto-charge threshold
- Falls back to configured threshold (75 EUR) if API doesn't provide it
- Smart balance alerts with European formatting

### 🎯 ATC Optimization Rules
- Prioritizes Add-to-Cart (ATC) generation
- Rewards ATC performance with budget boosts
- Fast-tracks high-ATC ads to validation

### 🛡️ Enhanced Safety Features
- Active ads filtering (only shows ACTIVE status)
- Comprehensive account health monitoring
- Payment failure detection and alerts

## 📁 Project Structure

```
dean/
├── src/                    # Core automation logic
│   ├── main.py           # Main entry point
│   ├── meta_client.py   # Meta API client
│   ├── rules.py         # Business logic engine
│   ├── slack.py         # Slack notifications
│   └── stages/          # Stage-specific modules
├── config/              # Configuration files
│   ├── settings.yaml    # Main settings
│   └── rules.yaml       # Business rules
├── scripts/             # Setup and utility scripts
├── docs/                # Comprehensive documentation
└── data/                # SQLite databases and logs
```

## 🔍 Monitoring & Alerts

### Slack Integration
- **Run Summaries**: Performance metrics with European formatting
- **Stage Notifications**: Individual stage results
- **Health Alerts**: Account status and payment issues
- **Balance Warnings**: Dynamic threshold-based alerts

### Health Monitoring
- **Account Health**: Payment status, balance, spend caps
- **Performance Tracking**: CPA, ROAS, CTR monitoring
- **Data Quality**: Tracking pixel and conversion monitoring

## 📚 Documentation

- **[Complete Documentation](docs/README.md)** - Comprehensive setup and usage guide
- **[Account Health Monitoring](docs/ACCOUNT_HEALTH_MONITORING.md)** - Safety and monitoring features
- **[Rate Limiting Guide](docs/RATE_LIMITING.md)** - Advanced API rate limiting
- **[Configuration Guide](docs/CONFIGURATION.md)** - Detailed configuration options

## 🚨 Troubleshooting

### Common Issues

**API Rate Limits**: See [Rate Limiting Guide](docs/RATE_LIMITING.md)
**Missing Environment Variables**: Check `.env` file configuration
**Database Issues**: Ensure no concurrent runs
**Slack Failures**: Verify webhook URLs and permissions

### Debug Mode

```bash
# Enable detailed logging
export DEBUG=true
python src/main.py --simulate --explain
```

## 🔧 Advanced Features

### Queue Management
- **Supabase Integration**: Cloud-based creative queue
- **CSV/Excel Support**: Local file-based queue
- **Smart Launch**: Timezone-optimized ad launches

### Rate Limiting
- **Comprehensive API Limits**: All Meta Marketing API rate limits
- **Intelligent Retry**: Exponential backoff with tier-specific waits
- **Real-time Monitoring**: Rate limit usage tracking

### Safety Guardrails
- **Account Protection**: Prevents account suspensions
- **Budget Controls**: Daily limits and fairness rules
- **Emergency Stops**: Automatic pause for critical issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

This project is proprietary software. All rights reserved.

---

**Dean** - Intelligent advertising automation for the modern marketer.

*For detailed documentation, see [docs/README.md](docs/README.md)*
