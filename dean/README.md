# 🤖 Dean - ML-Enhanced Meta Ads Automation

Dean is a production-ready automation system for Meta Advantage+ Shopping campaigns. The runner combines deterministic guardrails with a deep ML stack that learns from Supabase-hosted performance data, refreshes creatives, and keeps humans in the loop via Slack.

## 🚀 Core Capabilities

- **Automation Runner** – `src/main.py` orchestrates Advantage+ campaign checks, applies safety guardrails, and coordinates Slack reporting.
- **Unified ML Pipeline** – `ml/ml_pipeline.py`, `ml/ml_intelligence.py`, and `ml/ml_decision_engine.py` handle training, caching, inference, and explainable kill/promote/scale decisions.
- **Advanced Enhancements** – `ml/ml_enhancements.py` & `ml/ml_advanced_features.py` add validation, anomaly detection, reinforcement learning, LSTM forecasting, SHAP explainability, and portfolio/budget optimisation.
- **Creative Intelligence** – `creative/image_generator.py`, `creative/advanced_creative.py`, and `ml/creative_pipeline.py` generate and score new assets using Flux + template libraries.
- **Analytics & Monitoring** – `analytics/performance_tracking.py`, `analytics/table_monitoring.py`, and `ml/ml_monitoring.py` track fatigue, health, and table integrity.
- **Infrastructure & Integrations** – Supabase storage (`infrastructure/supabase_storage.py`, `data_validation.py`, `transactions.py`), Meta & Flux API clients (`integrations/meta_client.py`, `integrations/flux_client.py`), Slack alerts, caching, scheduling, and rate-limit protection.

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────┐
│            Advantage+ Shopping Ads            │
└──────────────────────────────┬────────────────┘
                               │ Graph API
                               ↓
┌───────────────────────────────────────────────┐
│          Automation & Decision Layer           │
│  • Runner & scheduler (`src/main.py`)          │
│  • ASC+ stage logic (`stages/asc_plus.py`)     │
│  • Slack insights (`integrations/slack.py`)    │
└──────────────┬──────────────┬─────────────────┘
               │              │
               │              │
        ┌──────▼──────┐ ┌─────▼─────┐
        │ ML Pipeline │ │ Creative  │
        │ (`ml/*`)    │ │ Engine    │
        └──────┬──────┘ └─────┬─────┘
               │              │
               ▼              ▼
        ┌────────────────────────────┐
        │   Supabase Data Backbone   │
        │ performance_metrics,       │
        │ ad_lifecycle, time_series, │
        │ creative_intelligence, ... │
        └────────────────────────────┘
```

## 🆕 Recent Updates (November 2025)

- **ASC+ First Class Citizen** – `stages/asc_plus.py` consolidates rule logic, ML feedback, creative refresh, and lifecycle logging for a five-creative Advantage+ stack.
- **ML Reliability Hardening** – `ml/ml_pipeline.py` gained retry/backoff, execution telemetry, creative similarity cold starts, and weekly validation via `ModelValidator`.
- **Creative Automation Revamp** – Flux-powered generator + template libraries now run through `ml/advanced_system.py` for DNA analysis, variant testing, and self-healing refresh strategies.
- **Analytics Layer Expansion** – `analytics/performance_tracking.py` & `ml/ml_monitoring.py` capture fatigue velocity, momentum, and table health, feeding Slack learning summaries.
- **Infrastructure Safeguards** – `infrastructure/data_validation.py`, `rate_limit_manager.py`, `performance_optimization.py`, and `transactions.py` protect writes, Supabase quotas, and long-running jobs.

## ⚙️ Quick Start

1. **Install**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Dean.git
   cd Dean/dean
   pip install -r requirements.txt
   ```
2. **Configure environment**
   ```bash
   cp .env.example .env  # create manually if missing
   ```
   Required keys: `FB_APP_ID`, `FB_APP_SECRET`, `FB_ACCESS_TOKEN`, `FB_AD_ACCOUNT_ID`, `FB_PIXEL_ID`, `FB_PAGE_ID`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SLACK_WEBHOOK_URL`, `STORE_URL`.
3. **Set campaign & guardrails**  
   Edit `config/settings.yaml` (campaign IDs, budgets) and `config/rules.yaml` (CPA/ROAS thresholds).
4. **Provision Supabase tables**  
   Use the schema in `models/registry/` or copy the definitions from the Supabase dashboard exports referenced in the modules.
5. **Run**
   ```bash
   python src/main.py --profile production        # live
   python src/main.py --dry-run                   # inspect decisions
   python src/main.py --no-ml                     # legacy rule mode
   python src/main.py --explain                   # include reasoning
   ```

## 🔍 Key Modules

| Location | Purpose |
| --- | --- |
| `src/main.py` | Main automation runner, scheduler management, Slack reporting. |
| `src/stages/asc_plus.py` | Single-stage Advantage+ campaign handler with ML-aware guardrails. |
| `src/ml/ml_pipeline.py` | Orchestrates ML intelligence, anomaly checks, validation, and decision output. |
| `src/ml/ml_enhancements.py` | Model validation, data readiness tracking, anomaly detection, forecasting, similarity, causal analysis. |
| `src/ml/ml_advanced_features.py` | Reinforcement learning agent, LSTM predictor, SHAP explainability, portfolio optimisation, seasonal tuning. |
| `src/ml/advanced_system.py` | Aggregates creative DNA, variant testing, prompt evolution, budget optimisation, and self-healing automation. |
| `src/creative/image_generator.py` | Flux-powered static creative generator with prompt engineering safeguards. |
| `src/analytics/performance_tracking.py` | Fatigue detection, decay modelling, Supabase persistence. |
| `src/infrastructure/*` | Validation, caching, optimisation, rate limiting, Supabase storage, background jobs. |
| `src/integrations/*` | Meta Graph API, Flux API, Slack notifications, formatting helpers. |

## 📡 Automation Loop

1. Pull Meta insights for active Advantage+ ads.
2. Normalise metrics, sync lifecycle data, and append daily digests.
3. Run ML pipeline: detect anomalies, score performance, predict future outcomes, validate models, and assemble reasoning.
4. Apply ASC+ guardrails with ML overrides where confident; queue kills/promotes/scales.
5. Refresh creatives when fatigue or health signals require new assets.
6. Persist metrics, predictions, and creative intelligence to Supabase.
7. Push Slack updates with status, confidence, anomalies, and learning progress.

## 🧠 ML Insights

- **Confidence & Explainability** – Ensemble variance + SHAP (`ml/ml_advanced_features.py`) surface why a decision was made.
- **Time-Series Forecasting** – `ml/time_series_forecast.py` and LSTM predictors estimate CPA/ROAS trajectories.
- **Cold Start Strategy** – `CreativeSimilarityAnalyzer` scores new ads based on embeddings from proven winners.
- **Self-Healing** – `ml/auto_optimization.py` and `performance_adaptation.py` adjust thresholds and retry failed operations without manual intervention.

## 🗃 Supabase Data Contract

- `performance_metrics`, `time_series_data`, `fatigue_analysis`, `performance_decay` – used by analytics layer.
- `ad_lifecycle`, `ad_creation_times`, `creative_intelligence`, `ml_predictions`, `ml_models`, `model_validations` – consumed by ML pipeline and Slack digests.
- `models/registry/` contains reference exports; each write is validated through `infrastructure/data_validation.py` and `transactions.py`.

## 🔔 Monitoring & Alerts

- Slack notifications: run summaries, kill/promote alerts, budget warnings, and ML learning status (`integrations/slack.py`).
- Health checks: Supabase latency, rate limits, cache saturation (`infrastructure/health_check.py`, `rate_limit_manager.py`).
- Table watchdogs: `analytics/table_monitoring.py` raises alarms on missing rows or schema drift.

## 🧪 Troubleshooting

- **“ML system not available”** – ensure Supabase credentials and optional dependencies (xgboost, torch, shap, featuretools) are installed.
- **Timezone or date parsing errors** – `infrastructure/data_validation.py` normalises timestamps; confirm Meta account timezone via `ACCOUNT_TZ`.
- **Supabase rate limits** – batch writes through `infrastructure/transactions.py` and enable caching in `infrastructure/caching.py`.
- **Creative generation failures** – check Flux API keys and file quotas in `creative/image_generator.py`; fall back to templates when Flux unavailable.
- Use `python src/main.py --dry-run --explain` for verbose reasoning and anomaly diagnostics.

## 📦 Deployment

- **GitHub Actions** – schedule hourly runs with secrets for Meta, Supabase, Slack.
- **VPS / cron** – run every 30–60 minutes, rotating logs and using screen/tmux if background mode unavailable.
- **Background scheduler** – `infrastructure/scheduler.py` can keep the process alive locally; remember to stop it via `stop_background_scheduler()` on exit.

## 📚 Requirements

- Python 3.9+ (3.13 tested via bundled `venv/`)
- Core packages: `xgboost`, `scikit-learn`, `pandas`, `numpy`, `torch` (optional), `optuna`, `featuretools`, `supabase`, `slack_sdk`
- Hardware: 1 vCPU / 1GB RAM recommended for hourly runs with ML enabled.

## 🤝 Contributing

Dean ships as a turnkey system—fork and tailor rules, creatives, and ML components for your account. Please follow:

- Type hints + docstrings for new modules.
- Defensive error handling (no bare excepts).
- Validation on all external data writes.
- Update this README when introducing new subsystems.

---

**Dean** — self-learning Meta Advantage+ automation with auditable ML decisions and automated creative refresh.
