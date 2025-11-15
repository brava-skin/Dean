# 🤖 Dean - Enterprise Meta Ads Automation

Dean is a production-ready automation system for Meta Advantage+ Shopping campaigns. The system combines rule-based automation with a creative engine for intelligent ad management.

## 🚀 Core Capabilities

- **Rule-Based Automation** – Threshold-based decision making for ad management
- **Creative Engine** – AI-powered creative generation using Flux and OpenAI
- **ASC+ Campaign Management** – Automated Advantage+ Shopping Campaign optimization
- **Performance Tracking** – Comprehensive metrics collection and analysis
- **Auto-Refill Logic** – Automatic creative generation when active count drops
- **Slack Integration** – Real-time notifications and alerts

## 📋 Requirements

- Python 3.9+
- ffmpeg (system binary, not Python package)
- Meta Ads API credentials
- Supabase account
- OpenAI API key (for creative prompts)
- Flux API key (for image generation)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd dean
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

4. **Install ffmpeg:**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Linux
   sudo apt-get install ffmpeg
   ```

5. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## 🏃 Running

**Run Dean automation:**
```bash
make run
# or
python src/main.py
```

**Run with options:**
```bash
python src/main.py --stage asc_plus --dry-run
```

## 📁 Project Structure

```
dean/
├── src/                    # Source code
│   ├── main.py            # Entry point
│   ├── config.py          # Configuration constants
│   ├── stages/            # Campaign stages
│   │   └── asc_plus.py    # ASC+ campaign logic
│   ├── rules/             # Business rules
│   │   └── rules.py       # Rule engine
│   ├── creative/          # Creative services
│   │   ├── image_generator.py
│   │   ├── creative_intelligence.py
│   │   └── advanced_creative.py
│   ├── integrations/      # External integrations
│   │   ├── meta_client.py # Meta API
│   │   ├── flux_client.py # Flux API
│   │   └── slack.py       # Slack notifications
│   ├── infrastructure/    # Infrastructure layer
│   │   ├── storage.py     # Data storage
│   │   ├── caching.py     # Caching
│   │   ├── scheduler.py   # Task scheduling
│   │   ├── error_handling.py
│   │   ├── health_check.py
│   │   ├── optimization.py
│   │   ├── rate_limit_manager.py
│   │   ├── supabase_storage.py
│   │   ├── data_validation.py
│   │   ├── data_optimizer.py
│   │   ├── creative_storage.py
│   │   └── utils.py        # Utilities
│   └── analytics/         # Analytics
│       └── metrics.py     # Metrics collection
├── config/                # Configuration files
│   ├── production.yaml
│   ├── rules.yaml
│   └── settings.yaml
├── data/                  # Runtime data
├── scripts/               # Utility scripts
├── requirements.txt       # Dependencies
├── setup.py               # Package setup
├── pyproject.toml          # Modern Python config
└── Makefile              # Build automation
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Point                           │
│                  src/main.py (CLI)                       │
└──────────────────────┬────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│   Stages     │ │  Rules   │ │  Creative   │
│  (ASC+)      │ │  Engine  │ │  Engine     │
└──────┬───────┘ └────┬─────┘ └──────┬──────┘
       │              │               │
       └──────────────┼───────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌─────────────┐ ┌──────────┐ ┌─────────────┐
│ Integrations│ │Infrastructure│ Analytics │
│  (Meta,     │ │ (Storage,   │ (Metrics)  │
│   Flux,     │ │  Caching,   │            │
│   Slack)    │ │  Utils)     │            │
└─────────────┘ └──────────┘ └─────────────┘
```

### Core Components

1. **Entry Point** (`src/main.py`) - Orchestrates automation cycle, manages configuration, coordinates stage execution
2. **Stages** (`src/stages/`) - ASC+ campaign management with rule application, creative generation, and auto-refill
3. **Rules Engine** (`src/rules/`) - Business rule evaluation, threshold-based decisions, kill/promote/scale logic
4. **Creative Engine** (`src/creative/`) - Image generation via Flux, prompt engineering via OpenAI, performance tracking
5. **Integrations** (`src/integrations/`) - Meta Ads API, Flux API, Slack notifications
6. **Infrastructure** (`src/infrastructure/`) - Storage, caching, scheduling, error handling, health checks, rate limiting
7. **Analytics** (`src/analytics/`) - Metrics collection, performance tracking, data validation

### Data Flow

1. Configuration Loading → Load YAML configs and environment variables
2. Client Initialization → Initialize Meta, Flux, and Supabase clients
3. Stage Execution → Run ASC+ stage logic
4. Rule Evaluation → Apply business rules to ad performance
5. Creative Generation → Generate new creatives when needed
6. Data Storage → Store performance metrics in Supabase
7. Reporting → Send Slack notifications and generate reports

## 🧪 Development

**Install development dependencies:**
```bash
make install-dev
```

**Format code:**
```bash
make format
# or
black src/
```

**Lint code:**
```bash
make lint
# or
mypy src/
ruff check src/
```

**Clean build artifacts:**
```bash
make clean
```

### Adding New Features

1. **New Stage:** Create file in `src/stages/`, implement stage function, add to main.py
2. **New Rule:** Extend `src/rules/rules.py`, add rule configuration to `config/rules.yaml`
3. **New Integration:** Create client in `src/integrations/`, add to `__init__.py` exports
4. **New Infrastructure:** Add module to `src/infrastructure/`, export from `__init__.py`

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for public APIs
- Keep functions focused and small
- Use dataclasses where appropriate

## 🔧 Configuration

Configuration is managed through YAML files in `config/`:

- `production.yaml` - Production settings
- `rules.yaml` - Business rules and thresholds
- `settings.yaml` - General settings

Environment variables override YAML configuration. See `.env.example` for available options.

### Required Environment Variables

- `FB_APP_ID`, `FB_APP_SECRET`, `FB_ACCESS_TOKEN`, `FB_AD_ACCOUNT_ID`
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_TABLE`
- `OPENAI_API_KEY`, `FLUX_API_KEY`
- `SLACK_WEBHOOK_URL`
- `BREAKEVEN_CPA`, `COGS_PER_PURCHASE`, `USD_EUR_RATE`

## 🚀 Deployment

### Production Deployment

1. Set up virtual environment and install dependencies
2. Configure all environment variables
3. Verify configuration: `python src/main.py --dry-run`
4. Run in production: `python src/main.py`

### GitHub Actions

The project includes a GitHub Actions workflow (`.github/workflows/dean-automation.yml`) that runs Dean on a schedule.

**Setup:**
1. Set up GitHub Secrets with all required environment variables
2. Update workflow file to set `AUTOMATION_PAUSED: "false"`
3. Workflow runs every 30 minutes by default

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run Dean
CMD ["python", "src/main.py"]
```

Build and run:
```bash
docker build -t dean .
docker run --env-file .env dean
```

## 🔍 Troubleshooting

### Common Issues

1. **ffmpeg not found:**
   - Install ffmpeg system package
   - Verify PATH includes ffmpeg

2. **API rate limits:**
   - Check rate limit configuration
   - Review `META_REQUEST_DELAY` setting

3. **Supabase connection errors:**
   - Verify credentials
   - Check network connectivity
   - Review Supabase logs

4. **Creative generation failures:**
   - Verify Flux API key
   - Check OpenAI API key
   - Review API quotas

## 🔐 Security

- Never commit `.env` files
- Use environment variables for sensitive data
- Rotate API keys regularly
- Review Supabase RLS policies

## 📝 License

MIT License

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## 📞 Support

For issues and questions, please open an issue in the repository.
