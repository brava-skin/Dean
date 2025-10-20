# Dean - Meta Ads Automation Platform

> **Intelligent advertising automation for Meta (Facebook) advertising platforms**

Dean is a production-ready automation system that manages the entire lifecycle of ad creatives from testing through validation to scaling, with intelligent decision-making based on performance metrics.

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

```bash
# Run all stages
python src/main.py

# Run specific stage
python src/main.py --stage testing
python src/main.py --stage validation
python src/main.py --stage scaling

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
