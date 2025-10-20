# Dean - Brava Skin Ads Automation

A production-ready automation system for continuous creative testing, validation, and scaling on Meta (Facebook) advertising platforms. Dean automates the entire lifecycle of ad creatives from initial testing through validation to scaling, with intelligent decision-making based on performance metrics.

## 🚀 Overview

Dean is a sophisticated advertising automation platform that:

- **Tests** new creative assets with controlled budgets and performance thresholds
- **Validates** promising creatives with extended testing periods and stricter criteria  
- **Scales** winning creatives with intelligent budget allocation and portfolio management
- **Monitors** account health with comprehensive guardrails and safety nets
- **Reports** performance through Slack notifications and detailed logging

## 🏗️ Architecture

The system follows a three-stage pipeline:

```
Creative Queue → Testing → Validation → Scaling
     ↓             ↓         ↓          ↓
   Supabase/CSV   Budget    Extended   Portfolio
   File Input     Control   Testing    Management
```

### Core Components

- **Main Runner** (`main.py`): Orchestrates the entire automation pipeline
- **Meta Client** (`meta_client.py`): Handles all Facebook/Meta API interactions
- **Rule Engine** (`rules.py`): Implements business logic and decision rules
- **Storage** (`storage.py`): Manages SQLite state and logging
- **Stages**: Specialized modules for each automation phase
  - `testing.py`: New creative testing with budget controls
  - `validation.py`: Extended validation with performance thresholds
  - `scaling.py`: Advanced scaling with portfolio management

## 📋 Features

### Testing Stage
- **Advanced Learning Acceleration Rules**: 7-tier performance-based system
  - **Tier 1**: Multi-ATC ads (3+ ATCs) get €500 budget for extended learning
  - **Tier 2**: High-ATC ads (2+ ATCs) get €400 budget with fast-track to validation
  - **Tier 3**: Single ATC ads get €300 budget with learning acceleration
  - **Tier 4**: High-CTR ads (CTR > 2%) get €250 budget for learning
  - **Tier 5**: Good CTR ads (CTR > 1.5%) get €200 budget
  - **Tier 6**: Decent CTR ads (CTR > 1%) get €160 budget
  - **Tier 7**: Poor performance ads killed at €90 (low CTR, no ATC)
- **ATC Optimization Rules**: Prioritizes Add-to-Cart generation
  - Reward ANY ATC with €50 budget boost
  - Fast-track 2+ ATC ads to validation stage
  - ATC learning acceleration with extended budgets
- **Zero-Performance Quick Kill**: Ads with CTR < 0.1% killed at €30 to save budget
- Automated creative launch from queue (Supabase or CSV)
- Budget control with daily limits and fairness rules
- Smart budget reallocation from poor to high performers
- Queue rotation and launch deferral
- Instagram and Facebook placement optimization

### Validation Stage  
- **Tiered Validation Rules**: 7-tier system mirroring testing tiers
  - **Tier 1**: Strong ATC signals (3+ ATCs) get €500 budget with extended learning
  - **Tier 2**: High ATC performance (2+ ATCs) get €400 budget
  - **Tier 3**: Excellent performance (high CTR + ATC) get €300 budget
  - **Tier 4**: High CTR learning (no ATC yet) get €240 budget
  - **Tier 5**: Good performance (CTR > 1% + ATC) get €200 budget
  - **Tier 6**: Decent CTR (no ATC) get €160 budget
  - **Tier 7**: Poor performance killed at €90 (low CTR, no ATC)
- Extended testing with higher budget allocation
- Stricter performance requirements for promotion
- Multi-day stability requirements
- Soft pass options for borderline performers

### Scaling Stage
- **Tiered Scaling Rules**: Aligned with testing tiers, stricter at scale
  - **Zero-performance safeguard**: Runaway spend protection at €80 (CTR < 0.1%)
  - **Poor engagement at scale**: Low CTR + no ATC killed at €200
  - **Core scaling rules**: CPA > €40 for 2+ days, ROAS < 1.2 for 3+ days
  - **Spend protection**: No purchase after €150 spend
- Intelligent budget scaling with hysteresis protection
- Portfolio management and reinvestment strategies
- Creative duplication for high performers
- CBO (Campaign Budget Optimization) support
- Advanced pacing controls and velocity monitoring

### Safety & Monitoring
- Account-level guardrails and CPA monitoring
- Spend velocity controls and pacing freezes
- Emergency stop mechanisms
- Comprehensive health checks
- Data quality monitoring and alerts

## 🆕 Recent Updates

### 🎯 7-Tier Performance System
- **Testing Stage**: 7-tier system with ATC prioritization
  - Multi-ATC ads (3+ ATCs) get €500 budget for extended learning
  - High-ATC ads (2+ ATCs) get €400 budget with fast-track to validation
  - Single ATC ads get €300 budget with learning acceleration
  - High-CTR ads (CTR > 2%) get €250 budget for learning
  - Good CTR ads (CTR > 1.5%) get €200 budget
  - Decent CTR ads (CTR > 1%) get €160 budget
  - Poor performance ads killed at €90 (low CTR, no ATC)
- **Validation Stage**: Mirroring tiered rules with stage-appropriate budgets
- **Scaling Stage**: Aligned tiers with stricter scaling thresholds

### 🛒 ATC Optimization Rules
- **ATC Reward System**: Reward ANY ATC with €50 budget boost
- **Fast-track Validation**: 2+ ATC ads promoted to validation stage
- **ATC Learning Acceleration**: Extended budgets for ATC-generating ads
- **Zero-Performance Quick Kill**: CTR < 0.1% killed at €30 to save budget

### 🔄 Dynamic Billing Threshold Detection
- Automatically detects Meta's current auto-charge threshold
- Falls back to configured threshold (75 EUR) if API doesn't provide it
- Smart balance alerts with European formatting
- Shows threshold source (Meta API vs configured fallback)

### 🛡️ Enhanced Safety Features
- Active ads filtering (only shows ACTIVE status)
- Comprehensive account health monitoring
- Payment failure detection and alerts
- European number formatting in all reports

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Meta Business Account with API access
- Slack workspace (for notifications)
- Optional: Supabase account (for queue management)

### Setup

**Quick Setup (macOS):**
```bash
git clone <repository-url>
cd dean
chmod +x scripts/setup_macos.sh
./scripts/setup_macos.sh
```

**Manual Setup:**
1. **Clone and install dependencies:**
```bash
cd dean
pip install -r requirements.txt
```

2. **Environment Configuration:**
Create a `.env` file with required variables:

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
SLACK_WEBHOOK_URL_ERRORS=https://hooks.slack.com/services/...

# Optional: Supabase (for queue management)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives

# Optional: Economics
BREAKEVEN_CPA=34
COGS_PER_PURCHASE=15
USD_EUR_RATE=0.92
```

3. **Configuration Files:**
- `config/settings.yaml`: Main configuration
- `config/rules.yaml`: Business rules and thresholds

## 🚀 Usage

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

### Advanced Options

```bash
# Custom configuration files
python src/main.py --settings config/custom_settings.yaml --rules config/custom_rules.yaml

# Profile-based execution
python src/main.py --profile staging
python src/main.py --profile production

# Simulation with date range
python src/main.py --simulate --since 2024-01-01 --until 2024-01-31

# Explain mode (show decisions without acting)
python src/main.py --explain
```

### Scheduling

For production use, schedule the automation to run regularly:

```bash
# Cron example - run every 2 hours
0 */2 * * * cd /path/to/dean && python src/main.py --profile production
```

## ⚙️ Configuration

### Settings Structure

The main configuration is in `config/settings.yaml`:

```yaml
# Account Configuration
ids:
  testing_campaign_id: "120231838265440160"
  testing_adset_id: "120231838265460160"
  validation_campaign_id: "120231838417260160"
  scaling_campaign_id: "120231838441470160"

# Testing Stage
testing:
  daily_budget_eur: 50
  keep_ads_live: 4
  max_active_ads: 4
  minimums:
    min_impressions: 300
    min_spend_eur: 10

# Validation Stage  
validation:
  adset_budget_eur: 40
  min_days: 2
  max_days: 3
  minimums:
    min_impressions: 600
    min_spend_eur: 40

# Scaling Stage
scaling:
  adset_start_budget_eur: 100
  allow_cbo: true
  budget:
    min_eur: 10
    max_eur: 5000
    max_step_pct: 100
```

### Rules Configuration

Business logic and thresholds are defined in `config/rules.yaml`:

```yaml
# Performance Thresholds
thresholds:
  cpa:
    testing_max: 36
    validation_max: 28
    scaling_kill_max: 40
  roas:
    testing_min: 1.5
    validation_min: 1.8
    scaling_kill_min: 1.2

# Testing Rules
testing:
  kill:
    - {type: "spend_no_purchase", spend_gte: 45}
    - {type: "ctr_below", ctr_lt: 0.008, spend_gte: 40}
    - {type: "cpa_gte_over_days", cpa_gte: 38, days: 3}
  advance:
    rules:
      - {type: "purchase_gte", purchases_gte: 1}
      - {type: "cpa_lte", cpa_lte: 36}
```

## 📊 Queue Management

### Supabase Integration (Recommended)

Dean can use Supabase as a queue source for creative management:

```sql
-- Example table structure
CREATE TABLE meta_creatives (
  id SERIAL PRIMARY KEY,
  video_id TEXT,
  filename TEXT,
  avatar TEXT,
  visual_style TEXT,
  script TEXT,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT NOW()
);
```

### CSV/Excel Files

Alternative queue management using local files:

```csv
video_id,filename,avatar,visual_style,script,status
1438715257185990,video1.mp4,avatar1.jpg,style1,script1,pending
1438715257185991,video2.mp4,avatar2.jpg,style2,script2,pending
```

## 🔧 API Reference

### Main Modules

#### `main.py`
The main entry point that orchestrates the automation pipeline.

**Key Functions:**
- `main()`: Main entry point with argument parsing
- `load_queue()`: Loads creative queue from file or Supabase
- `health_check()`: Validates system health before execution
- `run_stage()`: Executes individual automation stages

#### `meta_client.py`
Handles all Meta/Facebook API interactions with retry logic and error handling.

**Key Classes:**
- `MetaClient`: Main client for API operations
- `AccountAuth`: Authentication configuration
- `ClientConfig`: Client configuration settings

**Key Methods:**
- `get_ad_insights()`: Retrieves ad performance data
- `create_ad()`: Creates new ad creatives
- `update_adset_budget()`: Updates adset budgets
- `pause_ad()`: Pauses underperforming ads

#### `rules.py`
Implements business logic and decision rules for automation.

**Key Classes:**
- `RuleEngine`: Main rule evaluation engine
- `Metrics`: Performance metrics calculation
- `MetricsConfig`: Metrics configuration

**Key Methods:**
- `evaluate_kill_rules()`: Determines if ads should be killed
- `evaluate_advance_rules()`: Determines if ads should advance stages
- `calculate_metrics()`: Computes performance metrics

#### `storage.py`
Manages SQLite database for state persistence and logging.

**Key Classes:**
- `Store`: Main storage interface
- `LogEntry`: Log entry structure

**Key Methods:**
- `log()`: Records system events
- `get_state()`: Retrieves stored state
- `set_state()`: Updates stored state

### Stage Modules

#### `testing.py`
Handles the testing stage of new creative assets.

**Key Functions:**
- `run_testing_tick()`: Main testing stage execution
- `launch_new_ads()`: Launches new ads from queue
- `evaluate_testing_performance()`: Evaluates ad performance
- `kill_underperformers()`: Removes underperforming ads

#### `validation.py`
Manages the validation stage with extended testing.

**Key Functions:**
- `run_validation_tick()`: Main validation stage execution
- `promote_to_scaling()`: Promotes validated ads to scaling
- `evaluate_validation_performance()`: Validation performance evaluation

#### `scaling.py`
Handles advanced scaling with portfolio management.

**Key Functions:**
- `run_scaling_tick()`: Main scaling stage execution
- `scale_budgets()`: Intelligent budget scaling
- `duplicate_creatives()`: Creative duplication for winners
- `portfolio_reinvestment()`: Portfolio budget reallocation

## 🔍 Monitoring & Alerts

### Slack Integration

Dean sends comprehensive notifications to Slack channels:

- **Run Summaries**: Daily execution summaries with performance metrics
- **Stage Notifications**: Individual stage results and actions taken
- **Error Alerts**: System errors and API failures
- **Performance Alerts**: Account-level performance issues

### Logging

The system maintains detailed logs in multiple formats:

- **SQLite Database**: Structured logging in `data/state.sqlite`
- **JSON Logs**: Machine-readable logs in `data/actions.log.jsonl`
- **Daily Digests**: Summary logs in `data/digests/`

### Health Monitoring

Built-in health checks monitor:

- Database connectivity and performance
- Meta API availability and response times
- Slack notification delivery
- Account performance metrics
- Data quality and tracking issues
- **Ad Account Health**: Payment status, balance, spend caps, and account restrictions

### Ad Account Health Monitoring

The system now includes comprehensive ad account health monitoring to prevent account suspensions and payment issues:

- **Payment Monitoring**: Detects payment failures, expired cards, and declined transactions
- **Balance Alerts**: Warns when account balance is low or negative
- **Spend Cap Warnings**: Alerts when approaching daily/monthly spending limits
- **Account Status**: Monitors account restrictions and suspensions
- **Business Verification**: Ensures business information is complete

For detailed information, see [Account Health Monitoring Documentation](ACCOUNT_HEALTH_MONITORING.md).

### Advanced Rate Limiting
- **Comprehensive API Rate Limiting**: Implements all Meta Marketing API rate limiting types
- **API-Level Scoring**: Reads=1pt, Writes=3pts with tier-based limits (60pts dev/9000pts standard)
- **Business Use Case (BUC) Limits**: Endpoint-specific rate limiting with automatic headers
- **Ads Insights Platform Limits**: App-level rate limiting for insights queries
- **Budget Change Limits**: Ad account spend limits (10/day) and ad set budget limits (4/hour)
- **Intelligent Error Handling**: Handles all rate limit error codes with appropriate retry logic
- **Real-time Monitoring**: Track rate limit usage and status with detailed metrics
- **Automatic Retry Logic**: Exponential backoff with tier-specific wait times

For comprehensive rate limiting management, see [Rate Limiting Guide](RATE_LIMITING.md).

## 🚨 Troubleshooting

### Common Issues

**1. API Rate Limits**
```
Error: Rate limit exceeded
Solution: See comprehensive [Rate Limiting Guide](RATE_LIMITING.md) for advanced rate limiting management
```

**2. Missing Environment Variables**
```
Error: Missing required environment variables
Solution: Check .env file and ensure all required variables are set
```

**3. Database Lock Issues**
```
Error: Database is locked
Solution: Check for concurrent runs and ensure proper process management
```

**4. Slack Notification Failures**
```
Error: Slack webhook failed
Solution: Verify SLACK_WEBHOOK_URL and check Slack app permissions
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Enable debug logging
export DEBUG=true
python src/main.py --simulate --explain
```

### Health Checks

Run system health checks:

```bash
# Check system health
python -c "
from src.main import health_check
from src.storage import Store
from src.meta_client import MetaClient
# Run health check
"
```

## 📈 Performance Tuning

### Budget Optimization

- Adjust daily budgets in `settings.yaml`
- Configure minimum spend thresholds
- Set appropriate CPA and ROAS targets

### Queue Management

- Optimize queue size for testing velocity
- Configure launch deferral for timezone optimization
- Set appropriate fairness rules

### Scaling Parameters

- Tune scaling thresholds for your business model
- Configure portfolio management rules
- Set appropriate cooldown periods

## 🔒 Security

### API Security

- Use service accounts with minimal required permissions
- Rotate access tokens regularly
- Monitor API usage and costs

### Data Protection

- Encrypt sensitive configuration data
- Use secure storage for credentials
- Implement proper access controls

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Include type hints for all functions
- Add comprehensive docstrings
- Write tests for new functionality

## 📄 License

This project is proprietary software. All rights reserved.

## 📞 Support

For support and questions:

- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section

---

**Dean** - Intelligent advertising automation for the modern marketer.