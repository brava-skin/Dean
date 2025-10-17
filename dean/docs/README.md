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
- Automated creative launch from queue (Supabase or CSV)
- Budget control with daily limits and fairness rules
- Performance-based kill rules (CPA, ROAS, CTR thresholds)
- Queue rotation and launch deferral
- Instagram and Facebook placement optimization

### Validation Stage  
- Extended testing with higher budget allocation
- Stricter performance requirements for promotion
- Multi-day stability requirements
- Soft pass options for borderline performers

### Scaling Stage
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

## 🚨 Troubleshooting

### Common Issues

**1. API Rate Limits**
```
Error: Rate limit exceeded
Solution: Increase META_WRITE_COOLDOWN_SEC or reduce concurrency
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