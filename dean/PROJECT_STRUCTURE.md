# Dean Project Structure

## 📁 Clean Project Organization

```
dean/
├── 📁 config/                    # Configuration files
│   ├── production.yaml          # Production ML settings
│   ├── rules.yaml               # Business rules & thresholds
│   └── settings.yaml            # System settings
├── 📁 data/                      # Runtime data (SQLite)
│   ├── state.sqlite             # System state
│   ├── slack_outbox.sqlite      # Slack message queue
│   └── digests/                 # Historical data
├── 📁 docs/                     # Documentation
│   └── advanced/                # Advanced features
│       ├── ACCOUNT_HEALTH_MONITORING.md
│       └── RATE_LIMITING.md
├── 📁 scripts/                  # Setup scripts
│   └── setup_macos.sh          # macOS setup
├── 📁 src/                      # Source code
│   ├── main.py                 # 🚀 Main entry point
│   ├── stages/                 # Ad lifecycle stages
│   │   ├── testing.py          # Testing stage logic
│   │   ├── validation.py       # Validation stage logic
│   │   └── scaling.py          # Scaling stage logic
│   ├── ml_*.py                 # 🤖 ML Intelligence System
│   │   ├── ml_intelligence.py  # Core ML system
│   │   ├── ml_pipeline.py      # ML pipeline orchestration
│   │   ├── ml_status.py        # ML monitoring & diagnostics
│   │   └── ml_*.py             # Advanced ML features
│   ├── meta_client.py          # Meta Ads API client
│   ├── rules.py                # Business rules engine
│   ├── slack.py                # Slack notifications
│   └── utils.py                # Utilities
├── README.md                    # 📖 Main documentation
├── requirements.txt             # Python dependencies
├── supabase_schema.sql         # 🗄️ Database schema
└── supabase_security_fixes.sql # 🔒 Security optimizations
```

## 🎯 Key Components

### Core System
- **`main.py`** - Main entry point with ML-enhanced automation
- **`stages/`** - Ad lifecycle management (testing → validation → scaling)
- **`meta_client.py`** - Meta Ads API integration with rate limiting
- **`rules.py`** - Business rules and decision logic

### ML Intelligence System
- **`ml_intelligence.py`** - Core ML models and training
- **`ml_pipeline.py`** - ML decision orchestration
- **`ml_status.py`** - ML monitoring and diagnostics
- **`ml_*.py`** - Advanced ML features (20+ enhancements)

### Configuration & Data
- **`config/`** - YAML configuration files
- **`data/`** - SQLite databases for local state
- **`docs/`** - Essential documentation only

## 🚀 Usage

```bash
# Run with ML system (default)
python src/main.py --profile production

# Run without ML (legacy mode)
python src/main.py --no-ml

# Run in dry-run mode
python src/main.py --dry-run
```

## 📊 ML System Status

The ML system provides comprehensive diagnostics:
- 🧠 **LEARNING** - Models actively training
- ✅ **READY** - Models ready for predictions  
- ⏳ **INITIALIZING** - Building knowledge base
- ❌ **ERROR** - Issues detected

## 🔧 Clean Structure Benefits

- ✅ **No unused files** - Only essential components
- ✅ **Clear organization** - Logical file grouping
- ✅ **Minimal documentation** - Only what's needed
- ✅ **Production ready** - Optimized for deployment
