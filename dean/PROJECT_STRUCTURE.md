# Dean Project Structure

```
dean/
├── config/                    # Configuration files
│   ├── rules.yaml            # Business rules and thresholds
│   └── settings.yaml         # Main configuration
├── data/                      # Data storage
│   ├── copy_bank.json        # Creative copy templates
│   ├── *.sqlite             # Local databases
│   └── digests/             # Daily digest logs
├── docs/                     # Documentation
│   ├── API_REFERENCE.md      # API documentation
│   ├── CONFIGURATION.md      # Configuration guide
│   ├── GITHUB_SETUP.md       # GitHub Actions setup
│   ├── INDEX.md              # Main documentation
│   ├── INSTALLATION.md       # Installation guide
│   └── USAGE.md              # Usage guide
├── scripts/                  # Utility scripts
│   └── setup_supabase.py     # Supabase setup helper
├── src/                      # Source code
│   ├── main.py               # Main entry point
│   ├── meta_client.py        # Meta API client
│   ├── slack.py              # Slack notifications
│   ├── storage.py            # Data storage
│   ├── rules.py              # Business rules engine
│   ├── utils.py              # Utility functions
│   ├── metrics.py            # Metrics calculation
│   └── stages/               # Automation stages
│       ├── testing.py        # Testing stage
│       ├── validation.py     # Validation stage
│       └── scaling.py       # Scaling stage
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
├── ENVIRONMENT_SETUP.md      # Environment setup guide
└── PROJECT_STRUCTURE.md      # This file
```

## Key Directories

### `/config`
Contains all configuration files:
- `settings.yaml` - Main configuration with all settings
- `rules.yaml` - Business rules and decision logic

### `/data`
Contains all data files:
- `copy_bank.json` - Creative copy templates
- `*.sqlite` - Local SQLite databases for state and logs
- `digests/` - Daily digest logs in JSONL format

### `/docs`
Contains all documentation:
- API reference, configuration guides, installation instructions
- GitHub Actions setup guide

### `/scripts`
Contains utility scripts:
- `setup_supabase.py` - Helper to set up Supabase table schema

### `/src`
Contains all source code:
- `main.py` - Main entry point and orchestration
- `meta_client.py` - Meta API client wrapper
- `slack.py` - Slack notification system
- `storage.py` - Data storage and persistence
- `rules.py` - Business rules engine
- `stages/` - Individual automation stages

## Running the Project

### Local Development
```bash
cd dean
python src/main.py --profile production
```

### GitHub Actions
The project includes GitHub Actions workflows that run automatically:
- Located in `.github/workflows/`
- Runs every hour on GitHub servers
- Requires GitHub secrets to be configured

### Setup
1. Configure environment variables (see `ENVIRONMENT_SETUP.md`)
2. Set up Supabase: `python scripts/setup_supabase.py`
3. Run the automation: `python src/main.py`
