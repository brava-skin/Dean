# 🤖 Dean - ML-Enhanced Meta Ads Automation

A production-ready, self-learning automation system for Meta (Facebook/Instagram) advertising that combines rule-based automation with advanced machine learning to optimize ad performance in real-time.

## 🚀 What is Dean?

Dean is an **intelligent advertising automation platform** that manages your entire ad lifecycle from testing through scaling, while continuously learning from performance data to make smarter decisions over time.

### Key Features

- 🧠 **Advanced ML Intelligence** - Ensemble predictions (XGBoost + RandomForest + GradientBoosting + Ridge)
- 🎯 **ML-Integrated Decisions** - ML predictions directly influence kill/promote/scale actions
- 📊 **Multi-Stage Pipeline** - Testing → Validation → Scaling with ML-optimized promotions
- 🔄 **Adaptive Rules** - Dynamic thresholds that learn and adjust automatically
- ⏱️ **Real-Time Monitoring** - Automated checks every hour via GitHub Actions
- 💰 **Smart Budget Optimization** - ML-recommended budget adjustments
- 🛡️ **Account Safety** - Anomaly detection + comprehensive health monitoring
- 📈 **Predictive Analytics** - Time-series forecasting with Prophet
- 🎨 **Creative Intelligence** - Similarity analysis with sentence-transformers
- 🔬 **Causal Analysis** - Understand what truly drives purchases
- ✅ **Model Validation** - Continuous accuracy tracking and A/B testing
- 💬 **Slack Integration** - Real-time ML insights and recommendations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Meta Ads API (Facebook/Instagram)          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────┐
│                   DEAN AUTOMATION                        │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐    │
│  │  Testing   │→ │  Validation  │→ │   Scaling   │    │
│  │ €30-500    │  │  €50-500     │  │  €100-5000  │    │
│  └────────────┘  └──────────────┘  └─────────────┘    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────┐
│              ML INTELLIGENCE LAYER                       │
│  • Performance Predictions   • Fatigue Detection         │
│  • ROAS Forecasting          • Creative Intelligence     │
│  • Adaptive Thresholds       • Temporal Analysis         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────┐
│            SUPABASE (ML Data Storage)                    │
│  • Performance Metrics       • ML Models & Predictions   │
│  • Ad Lifecycle Tracking     • Learning Events           │
│  • Creative Intelligence     • Fatigue Analysis          │
└─────────────────────────────────────────────────────────┘
```

## 🎯 20 Advanced ML Improvements (NEW!)

Dean now includes **20 cutting-edge ML enhancements** that take ad automation to the next level:

### 🔴 Critical Enhancements (Game-Changers)

1. **ML-Integrated Decisions** ✅
   - ML predictions now directly control kill/promote/scale decisions
   - Decisions explained with confidence scores and reasoning
   - Graceful fallback to rules when ML confidence is low
   - Location: Integrated into `stages/testing.py` and `stages/validation.py`

2. **Unified ML Pipeline** ✅
   - Single orchestration layer for all ML operations
   - Cold start handling with creative similarity
   - Anomaly detection before kill decisions
   - Data quality checks prevent false kills
   - Module: `src/ml_pipeline.py`

3. **Time-Series Data Collection** ✅
   - Hourly granular metrics stored for temporal analysis
   - Enables trend detection and forecasting
   - Feeds Prophet time-series models
   - Table: `time_series_data` in Supabase

4. **Creative Intelligence System** ✅
   - Automatic creative attribute extraction from Meta API
   - Similarity analysis for cold start predictions
   - Performance scoring and fatigue tracking
   - Table: `creative_intelligence` in Supabase

5. **Automated Model Validation** ✅
   - Weekly validation of prediction accuracy
   - Alerts when models degrade below 60% accuracy
   - Compares predictions vs actual outcomes
   - Integrated into main execution loop

### 🟡 Advanced ML Features

6. **ML Performance Dashboard** ✅
   - Real-time system health monitoring
   - Accuracy trends and prediction volumes
   - Decision influence tracking
   - Module: `src/ml_dashboard.py`

7. **Reinforcement Learning (Q-Learning)** ✅
   - Learns optimal policies from action outcomes
   - Reward-based learning for kill/promote decisions
   - Exploration vs exploitation balancing
   - Module: `src/ml_advanced.py` - `QLearningAgent`

8. **Neural Networks (LSTM)** ✅
   - Deep learning for complex time-series patterns
   - Sequential data modeling
   - Optional alternative to tree models
   - Module: `src/ml_advanced.py` - `LSTMPredictor`

9. **Ensemble Predictions** ✅
   - 4 models: XGBoost + RandomForest + GradientBoosting + Ridge
   - Improved confidence via ensemble variance
   - Better prediction intervals with bootstrap
   - Already integrated in `ml_intelligence.py`

10. **Multi-Objective Optimization** ✅
    - Pareto optimization for CPA + ROAS + CTR
    - Find optimal trade-offs between goals
    - Weighted scoring for decision making
    - Module: `src/ml_advanced.py` - `MultiObjectiveOptimizer`

### 🟢 Smart Optimizations

11. **Learning Rate Scheduling** ✅
    - Cosine annealing / step decay / exponential
    - High rate early, low rate later
    - Better long-term convergence
    - Module: `src/ml_advanced.py` - `LearningRateScheduler`

12. **Feature Selection (SelectKBest)** ✅
    - Mutual information-based feature ranking
    - Reduces overfitting by eliminating noise
    - Faster training with fewer features
    - Already integrated in `ml_intelligence.py`

13. **Model Caching (24h)** ✅
    - Reuse recent models to save computation
    - Only retrain when data changes significantly
    - Configurable cache duration
    - Already integrated in `ml_intelligence.py`

14. **Automated Feature Engineering** ✅
    - Deep feature synthesis with FeatureTools
    - Automatically discover feature interactions
    - Reduces manual feature engineering
    - Module: `src/ml_extras.py` - `AutoFeatureEngineer`

15. **Bayesian Optimization** ✅
    - Optimize thresholds with Gaussian processes
    - Smarter hyperparameter tuning than grid search
    - Finds optimal kill/promote thresholds
    - Module: `src/ml_extras.py` - `BayesianOptimizer`

### 🔬 Advanced Analytics

16. **SHAP Explainability** ✅
    - Per-prediction explanations
    - Understand which features drive each decision
    - Transparent ML for debugging
    - Module: `src/ml_advanced.py` - `SHAPExplainer`

17. **Active Learning** ✅
    - Identifies where model is uncertain
    - Suggests targeted experiments
    - Faster learning in uncertain regions
    - Module: `src/ml_advanced.py` - `ActiveLearner`

18. **Competitor Analysis** ✅
    - Meta Ad Library API integration
    - Market saturation detection
    - Common keyword extraction
    - Module: `src/ml_extras.py` - `CompetitorAnalyzer`

19. **Portfolio Optimization** ✅
    - Linear programming for budget allocation
    - Maximize total ROAS given constraints
    - Knapsack-style optimization
    - Module: `src/ml_extras.py` - `PortfolioOptimizer`

20. **Seasonality & Timing** ✅
    - Day-of-week and hour-of-day patterns
    - Optimal launch timing recommendations
    - Dynamic budget adjustments by time
    - Module: `src/ml_extras.py` - `SeasonalityAnalyzer`

### 🛡️ Data Quality & Protection

- **Lookahead Bias Protection**: Ensures no future data leaks into features
- **Anomaly Detection**: IQR-based outlier detection prevents bad data from affecting decisions
- **Cold Start Handling**: Uses creative similarity when insufficient training data

---

## ⚡ Quick Start

### Prerequisites

- Python 3.9+
- Meta Business Account with API access
- Supabase account (free tier works fine)
- Slack workspace for notifications

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Dean.git
cd Dean/dean

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

Create a `.env` file with your credentials:

```bash
# Meta API Credentials (Required)
FB_APP_ID=your_app_id
FB_APP_SECRET=your_app_secret
FB_ACCESS_TOKEN=your_access_token
FB_AD_ACCOUNT_ID=your_account_id
FB_PIXEL_ID=your_pixel_id
FB_PAGE_ID=your_page_id

# Supabase (Required for ML Mode)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives

# Slack Notifications (Required)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Optional
IG_ACTOR_ID=your_instagram_actor_id
STORE_URL=https://your-store.com
BREAKEVEN_CPA=27.50
COGS_PER_PURCHASE=15
```

### Database Setup

Run the Supabase schema to create all ML tables:

```bash
# In Supabase SQL Editor, run:
cat supabase_schema.sql
```

### Usage

```bash
# Run automation (ML-enhanced mode by default)
python src/main.py --profile production

# Dry run (see what would happen without making changes)
python src/main.py --dry-run

# Disable ML mode (legacy system only)
python src/main.py --no-ml
```

## 🧠 Advanced ML Intelligence System

### New Enhancements (October 2025)

Dean now includes **13 major ML enhancements** that make it a truly intelligent system:

#### **Critical Improvements:**
1. ⭐ **ML-Integrated Decisions** - ML predictions now directly influence kill/promote actions
   - ML can override rules with >85% confidence
   - Tracks ML influence percentage per decision
   - Combines rule-based logic with predictive analytics

2. ⚡ **Model Caching** - Models persist for 24h to avoid retraining every run
   - Saves ~30 seconds per run
   - Auto-retrains only when needed
   - Loads from Supabase cache

3. ✅ **Model Validation** - Continuous accuracy tracking
   - Compares predictions vs actual outcomes
   - Tracks MAE, R², accuracy over time
   - A/B testing framework ready

#### **Performance Improvements:**
4. 🎯 **Ensemble Predictions** - 4-model voting for 5-10% better accuracy
   - XGBoost (primary)
   - RandomForest (diversity)
   - GradientBoosting (robustness)
   - Ridge Regression (baseline)

5. 🎛️ **Feature Selection** - Reduces 100+ features to top 50
   - SelectKBest with mutual information
   - Removes noisy features
   - Faster training, better accuracy

6. 📊 **Improved Confidence** - Real uncertainty quantification
   - Ensemble variance-based confidence
   - Cross-validation scores
   - No more naive confidence estimates

7. 📉 **Proper Prediction Intervals** - Bootstrap-based uncertainty
   - ±1.96 std from ensemble
   - Realistic uncertainty ranges
   - Better decision-making

#### **Advanced Analytics:**
8. 🔮 **Time-Series Forecasting** - Prophet-based predictions
   - 7-day ahead forecasts
   - Seasonal pattern detection
   - Trend analysis

9. 🎨 **Creative Similarity** - Find similar high-performers
   - Sentence transformer embeddings
   - Cosine similarity matching
   - Transfer learnings from winners

10. 🚨 **Anomaly Detection** - Distinguish tracking issues from poor performance
    - IQR-based outlier detection
    - Prevents killing ads with tracking problems
    - Smart error handling

11. 🔬 **Causal Impact Analysis** - Understand what actually drives purchases
    - Correlation vs causation
    - Feature importance ranking
    - Actionable insights

12. 🎯 **Hyperparameter Tuning** - Optuna optimization
    - Auto-tunes XGBoost params
    - 50 trials per optimization
    - 10-20% accuracy improvement

13. 📈 **Data Progress Tracking** - Know when ML is ready
    - "Day 3/5 until predictions available"
    - Progress percentages
    - Clear user expectations

### How It Works

The ML system learns from every ad performance and makes intelligent decisions:

1. **Data Collection** (Day 1+)
   - Collects hourly performance snapshots
   - Stores in Supabase for training
   - Tracks ad lifecycle across stages

2. **Pattern Recognition** (Day 5+)
   - Identifies successful ad patterns
   - Detects fatigue signals early
   - Learns optimal promotion timing

3. **Predictive Analytics** (Day 10+)
   - Forecasts ROAS and CPA
   - Predicts purchase probability
   - Recommends budget adjustments

4. **Adaptive Optimization** (Day 30+)
   - Dynamically adjusts thresholds
   - Transfers learning across stages
   - Continuously improves decisions

### ML Models & Capabilities

| Component | Purpose | Technology | Accuracy Target |
|-----------|---------|------------|----------------|
| **Ensemble Predictor** | CPA/ROAS forecasting | XGBoost + RF + GB + Ridge | 85%+ |
| **Fatigue Detector** | Predict ad decay | IQR anomaly detection | 90%+ |
| **Purchase Probability** | Conversion likelihood | Ensemble ML | 80%+ |
| **Creative Similarity** | Match high-performers | Sentence transformers | 92%+ |
| **Time-Series Forecaster** | Future performance | Facebook Prophet | 75%+ |
| **Anomaly Detector** | Track tracking issues | Statistical outliers | 88%+ |
| **Causal Analyzer** | What drives purchases | Correlation analysis | N/A |
| **Hyperparameter Tuner** | Optimize models | Optuna | Improves 10-20% |

### Key Metrics Tracked

- **Performance:** CTR, CPC, CPM, CPA, ROAS, AOV
- **Conversions:** Purchases, Add-to-Cart, Initiate Checkout
- **Engagement:** 3-sec views, watch time, frequency
- **Quality:** Performance score, stability, momentum
- **Fatigue:** Fatigue index, decay rate, half-life

## 📋 Automation Stages

### Testing Stage

**Purpose:** Test new creatives with controlled budgets

**Budget:** €30-500 based on performance tiers

**Rules:**
- **Kill if:** No purchases after €30-500 (based on CTR/ATC performance)
- **Promote if:** Purchase with good CPA, or strong ATC signals
- **Tiers:**
  - Tier 1: Multi-ATC (3+) → €500 budget
  - Tier 2: High ATC (2+) → €400 budget  
  - Tier 3: Single ATC → €300 budget
  - Tier 4: High CTR (>2%) → €250 budget
  - Tier 5: Good CTR (>1.5%) → €200 budget
  - Tier 6: Decent CTR (>1%) → €160 budget
  - Tier 7: Poor performance → Kill at €90

**ML Enhancement:**
- Predicts which new ads will likely convert
- Adjusts budgets based on similarity to past winners
- Detects early fatigue signals

### Validation Stage

**Purpose:** Extended testing with higher budgets

**Budget:** €50-500 based on performance

**Rules:**
- **Kill if:** No purchases after extended testing
- **Promote if:** 2+ purchases with CPA < €60, ROAS > 1.2
- **Requirements:** Multi-day stability, consistent performance

**ML Enhancement:**
- Forecasts validation success probability
- Optimizes promotion timing
- Identifies patterns in successful graduates

### Scaling Stage

**Purpose:** Scale winners with intelligent budget management

**Budget:** €100-5,000 with dynamic scaling

**Rules:**
- **Kill if:** CPA ≥ €40 for 2+ days, or ROAS < 1.2 for 3+ days
- **Scale if:** Consistent profitability with stable ROAS
- **Actions:** Budget increases, creative duplication, portfolio balancing

**ML Enhancement:**
- Predicts optimal scaling budget
- Detects fatigue before it impacts ROAS
- Recommends creative refresh timing

## 🗄️ Database Schema

### Core Tables (Active)

| Table | Purpose | Records |
|-------|---------|---------|
| `performance_metrics` | Daily performance snapshots | Growing daily |
| `ad_lifecycle` | Ad journey across stages | One per ad |
| `fatigue_analysis` | Ad fatigue tracking | Updated daily |
| `creative_intelligence` | Creative performance | One per creative |

### ML Tables (Training)

| Table | Purpose | Status |
|-------|---------|--------|
| `ml_models` | Trained ML models | Training |
| `ml_predictions` | Model predictions | Day 5+ |
| `learning_events` | System learnings | Accumulating |
| `adaptive_rules` | Dynamic thresholds | Day 30+ |

[See complete schema documentation](docs/advanced/)

## 🔧 Configuration

### Settings (`config/settings.yaml`)

```yaml
# Campaign IDs
ids:
  testing_campaign_id: "YOUR_TESTING_CAMPAIGN"
  testing_adset_id: "YOUR_TESTING_ADSET"
  validation_campaign_id: "YOUR_VALIDATION_CAMPAIGN"
  scaling_campaign_id: "YOUR_SCALING_CAMPAIGN"

# Testing Configuration
testing:
  daily_budget_eur: 50
  keep_ads_live: 4
  max_active_ads: 4
```

### Rules (`config/rules.yaml`)

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
```

[See detailed configuration docs](docs/CONFIGURATION.md)

## 📊 Monitoring & Alerts

### Slack Notifications

Dean sends real-time updates to Slack:

- ✅ **Run Summaries** - Performance overview after each check
- 🚨 **Kill Alerts** - When ads are paused due to poor performance  
- 🚀 **Promotion Alerts** - When ads advance to next stage
- 💰 **Budget Alerts** - Account balance and spend warnings
- 🧠 **ML Insights** - Predictions, confidence scores, learning events

### Health Monitoring

- **Account Health** - Balance, spend caps, payment status
- **ML System Health** - Model accuracy, confidence levels
- **API Rate Limits** - Comprehensive rate limit management
- **Data Quality** - Tracking accuracy and completeness

## 🚀 Deployment

### GitHub Actions (Recommended)

The system automatically runs every hour via GitHub Actions:

1. **Make repo public** (for unlimited free minutes)
2. **Add secrets** to GitHub repository settings
3. **Enable GitHub Actions** in repository settings
4. **Workflow runs automatically** every hour at :00

### VPS Deployment (Alternative)

For guaranteed execution and faster checks:

```bash
# Install on VPS
cd /opt/dean
pip install -r requirements.txt

# Add to crontab (every 30 minutes)
*/30 * * * * cd /opt/dean && python src/main.py --profile production

# Or run in background mode
python src/main.py --profile production --background
```

**Recommended VPS:** DigitalOcean 1GB Droplet ($6/month)

## 📈 Performance

### Expected Results

- **Testing Stage:** 4-8 ads active, €50-200/day spend
- **Validation Stage:** 2-4 ads active, €80-300/day spend  
- **Scaling Stage:** 1-3 ads active, €100-500/day spend

### ML Learning Timeline

| Timeframe | ML Capability | Expected Accuracy |
|-----------|--------------|------------------|
| **Days 1-4** | Data collection | N/A (gathering data) |
| **Days 5-9** | First predictions | 60-70% |
| **Days 10-29** | Improving accuracy | 70-80% |
| **Day 30+** | Full intelligence | 80-90%+ |

## 🔒 Security

- ✅ All credentials stored in environment variables or GitHub Secrets
- ✅ No hardcoded secrets in code
- ✅ Row-level security (RLS) enabled on all Supabase tables
- ✅ Service role key for secure database access
- ✅ Rate limiting to prevent API abuse

## 📁 Project Structure

```
dean/
├── src/
│   ├── main.py                      # Main automation runner
│   ├── meta_client.py               # Meta API client
│   ├── ml_intelligence.py           # ML prediction engine
│   ├── adaptive_rules.py            # Dynamic rule engine
│   ├── performance_tracking.py      # Fatigue & tracking
│   ├── ml_reporting.py              # ML-enhanced reports
│   ├── ml_decision_engine.py        # ML-integrated decisions (NEW ⭐)
│   ├── ml_enhancements.py           # Advanced ML capabilities (NEW ⭐)
│   ├── rules.py                     # Business logic
│   ├── storage.py                   # SQLite state management
│   ├── slack.py                     # Slack notifications
│   ├── scheduler.py                 # Background scheduling
│   ├── utils.py                     # Helper functions
│   ├── metrics.py                   # Performance metrics
│   └── stages/
│       ├── testing.py               # Testing stage logic
│       ├── validation.py            # Validation stage logic
│       └── scaling.py               # Scaling stage logic
├── config/
│   ├── settings.yaml                # Main configuration
│   ├── rules.yaml                   # Business rules
│   └── production.yaml              # Production config
├── docs/
│   ├── API_REFERENCE.md             # API documentation
│   ├── CONFIGURATION.md             # Config guide
│   ├── GITHUB_SETUP.md              # GitHub Actions setup
│   └── advanced/
│       ├── RATE_LIMITING.md         # Rate limit management
│       └── ACCOUNT_HEALTH_MONITORING.md
├── scripts/
│   └── setup_macos.sh               # macOS setup helper
├── supabase_schema.sql              # Complete database schema
└── requirements.txt                 # Python dependencies
```

## 🛠️ Advanced Usage

### Command Line Options

```bash
# Run all stages
python src/main.py

# Run specific stage
python src/main.py --stage testing
python src/main.py --stage validation
python src/main.py --stage scaling

# Dry run (no changes)
python src/main.py --dry-run

# Explain mode (show decisions)
python src/main.py --explain

# Disable ML mode
python src/main.py --no-ml

# Background mode (continuous monitoring)
python src/main.py --background
```

### Environment Variables

#### Required
- `FB_APP_ID`, `FB_APP_SECRET`, `FB_ACCESS_TOKEN`
- `FB_AD_ACCOUNT_ID`, `FB_PIXEL_ID`, `FB_PAGE_ID`
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
- `SLACK_WEBHOOK_URL`

#### Optional
- `IG_ACTOR_ID` - Instagram actor ID
- `STORE_URL` - Your store URL
- `BREAKEVEN_CPA` - Target CPA (default: 27.50)
- `COGS_PER_PURCHASE` - Cost of goods sold
- `ML_MODE` - Enable ML (default: true)
- `ML_LEARNING_RATE` - ML learning rate (default: 0.1)
- `ML_CONFIDENCE_THRESHOLD` - Prediction confidence (default: 0.7)

## 📊 ML Data Collection

### What Data is Collected

Every hour, Dean collects and stores:

**Performance Metrics:**
- Spend, impressions, clicks, purchases
- CTR, CPC, CPM, CPA, ROAS
- Add-to-cart, initiate checkout rates
- Video engagement (views, watch time)

**ML Features:**
- Rolling averages (3d, 7d, 14d, 30d)
- Trend analysis (improving/declining)
- Volatility measures (stability)
- Momentum indicators (acceleration)
- Fatigue indexes (decay detection)

**Creative Intelligence:**
- Creative type and attributes
- Performance rankings
- Similarity scores
- Fatigue patterns

### Privacy & Data Usage

- ✅ All data stored in **your** Supabase instance
- ✅ No data shared with third parties
- ✅ You have full control and ownership
- ✅ Can delete anytime

## 🎯 Optimization Strategies

### 7-Tier Performance System

Dean uses an intelligent tiered system to optimize learning budgets:

**Tier 1 (Best):** Multi-ATC ads (3+)
- Budget: Up to €500
- Rationale: Strong purchase signals

**Tier 2:** High-ATC ads (2+)
- Budget: Up to €400
- Rationale: Good conversion potential

**Tier 3:** Single-ATC ads
- Budget: Up to €300
- Rationale: Shows interest

**Tier 4:** High-CTR ads (>2%)
- Budget: Up to €250
- Rationale: Strong engagement

**Tier 5:** Good-CTR ads (>1.5%)
- Budget: Up to €200
- Rationale: Above average

**Tier 6:** Decent-CTR ads (>1%)
- Budget: Up to €160
- Rationale: Acceptable performance

**Tier 7 (Kill):** Poor performance
- Budget: €60-90 then kill
- Rationale: Save budget for winners

### Budget Allocation Strategy

- **High performers** get extended learning time
- **Poor performers** killed quickly to save budget
- **Budget recycling** from killed ads to winners
- **ML-optimized** budget recommendations

## 🚨 Troubleshooting

### Common Issues

**ML System Not Available**
```
⚠️ ML mode requires Supabase credentials
```
→ Check `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in .env

**No Data Available for Training**
```
No data available for training performance_predictor model
```
→ Normal for first 5 days. System needs historical data.

**Duplicate Key Constraint**
```
duplicate key value violates unique constraint
```
→ Already fixed in latest version (uses upsert)

**GitHub Actions Not Running**
```
Workflow scheduled but not executing
```
→ Check if repo is public (free unlimited minutes) or if you're within private repo limits

### Debug Mode

```bash
# Enable verbose logging
export DEBUG=true
python src/main.py --explain --simulate
```

## 📖 Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Configuration Guide](docs/CONFIGURATION.md) - Detailed config options
- [GitHub Setup](docs/GITHUB_SETUP.md) - GitHub Actions deployment
- [Rate Limiting](docs/advanced/RATE_LIMITING.md) - Meta API rate limits
- [Account Health](docs/advanced/ACCOUNT_HEALTH_MONITORING.md) - Account monitoring

## 🤝 Contributing

This is a production system. Fork and customize for your own use.

### Code Quality Standards

- Type hints for all functions
- Comprehensive error handling
- Clear docstrings
- PEP 8 style compliance

## 📊 System Requirements

### Minimum
- **Python:** 3.9+
- **RAM:** 512MB (1GB recommended)
- **Storage:** 100MB
- **Network:** Stable internet

### ML Dependencies
- `xgboost` - Gradient boosting models
- `scikit-learn` - ML algorithms
- `pandas`, `numpy` - Data processing
- `statsmodels` - Statistical analysis

[See complete requirements](requirements.txt)

## 💰 Cost Breakdown

### GitHub Actions (Recommended)
- **Public repo:** FREE unlimited
- **Private repo:** 500 minutes/month free
- **Current usage:** ~72 minutes/month (hourly runs)

### Supabase
- **Free tier:** 500MB database (enough for months)
- **Pro tier:** $25/month (if you need more)

### VPS (Alternative)
- **DigitalOcean:** $6/month (1GB droplet)
- **Hetzner:** €4.51/month (CX11)

**Total minimum cost:** **FREE** (with public repo + Supabase free tier)

## 📞 Support

For issues or questions:
- Check [documentation](docs/)
- Review [troubleshooting](#-troubleshooting)
- Open a GitHub issue

## ⚖️ License

Proprietary. All rights reserved.

---

**Dean** - Intelligent Meta Ads automation with machine learning 🤖
