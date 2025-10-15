# API Reference

This document provides detailed API documentation for all modules and functions in the Dean automation system.

## Core Modules

### main.py

The main entry point that orchestrates the automation pipeline.

#### Functions

##### `main() -> None`
Main entry point with argument parsing and orchestration.

**Parameters:** None

**Returns:** None

**Example:**
```python
if __name__ == "__main__":
    main()
```

##### `load_yaml(path: str) -> Dict[str, Any]`
Load YAML document or return empty dict on any error.

**Parameters:**
- `path` (str): Path to YAML file

**Returns:** Dict[str, Any]: Parsed YAML content or empty dict

**Example:**
```python
settings = load_yaml("config/settings.yaml")
```

##### `load_cfg(settings_path: str, rules_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]`
Load both settings and rules configuration files.

**Parameters:**
- `settings_path` (str): Path to settings YAML file
- `rules_path` (str): Path to rules YAML file

**Returns:** Tuple[Dict[str, Any], Dict[str, Any]]: Settings and rules dictionaries

**Example:**
```python
settings, rules = load_cfg("config/settings.yaml", "config/rules.yaml")
```

##### `load_queue(path: str) -> pd.DataFrame`
Load creatives queue from a file path (CSV/XLSX).

**Parameters:**
- `path` (str): Path to queue file

**Returns:** pd.DataFrame: Queue data with standardized columns

**Example:**
```python
queue_df = load_queue("data/creatives_queue.csv")
```

##### `load_queue_supabase(table: str = None, status_filter: str = "pending", limit: int = 64) -> pd.DataFrame`
Read creative rows from Supabase and normalize to expected columns.

**Parameters:**
- `table` (str, optional): Supabase table name
- `status_filter` (str): Status filter for rows
- `limit` (int): Maximum rows to return

**Returns:** pd.DataFrame: Queue data from Supabase

**Example:**
```python
queue_df = load_queue_supabase(table="meta_creatives", status_filter="pending")
```

##### `set_supabase_status(ids_or_video_ids: List[str], new_status: str, *, use_column: str = "id", table: str = None) -> None`
Generic status setter for meta_creatives.

**Parameters:**
- `ids_or_video_ids` (List[str]): List of IDs to update
- `new_status` (str): New status value
- `use_column` (str): Column to match against ("id" or "video_id")
- `table` (str, optional): Supabase table name

**Returns:** None

**Example:**
```python
set_supabase_status(["123", "456"], "launched", use_column="id")
```

##### `health_check(store: Store, client: MetaClient) -> Dict[str, Any]`
Validate system health before execution.

**Parameters:**
- `store` (Store): Database store instance
- `client` (MetaClient): Meta API client

**Returns:** Dict[str, Any]: Health check results

**Example:**
```python
health = health_check(store, client)
if not health["ok"]:
    print("Health check failed")
```

### meta_client.py

Handles all Meta/Facebook API interactions with retry logic and error handling.

#### Classes

##### `AccountAuth`
Authentication configuration for Meta API.

**Attributes:**
- `account_id` (str): Facebook ad account ID
- `access_token` (str): Long-lived access token
- `app_id` (str): Facebook app ID
- `app_secret` (str): Facebook app secret
- `api_version` (str, optional): API version to use

**Example:**
```python
auth = AccountAuth(
    account_id="act_123456789",
    access_token="your_token",
    app_id="your_app_id",
    app_secret="your_secret"
)
```

##### `ClientConfig`
Client configuration settings.

**Attributes:**
- `timezone` (str): Account timezone
- `currency` (str): Account currency
- `budgets` (Dict): Budget configuration
- `switches` (Dict): Feature switches

**Example:**
```python
config = ClientConfig(
    timezone="Europe/Amsterdam",
    currency="EUR"
)
```

##### `MetaClient`
Main client for Meta API operations.

**Parameters:**
- `accounts` (List[AccountAuth]): List of account authentications
- `cfg` (ClientConfig): Client configuration
- `store` (Store): Database store
- `dry_run` (bool): Whether to run in dry-run mode
- `tenant_id` (str): Tenant identifier

**Example:**
```python
client = MetaClient(
    accounts=[auth],
    cfg=config,
    store=store,
    dry_run=False
)
```

#### Methods

##### `get_ad_insights(level: str, fields: List[str], filtering: List[Dict] = None, time_range: Dict = None, action_attribution_windows: List[str] = None, paginate: bool = True) -> List[Dict[str, Any]]`
Retrieve ad performance insights from Meta API.

**Parameters:**
- `level` (str): Insight level ("ad", "adset", "campaign")
- `fields` (List[str]): Fields to retrieve
- `filtering` (List[Dict], optional): Filters to apply
- `time_range` (Dict, optional): Time range for data
- `action_attribution_windows` (List[str], optional): Attribution windows
- `paginate` (bool): Whether to paginate results

**Returns:** List[Dict[str, Any]]: Insight data

**Example:**
```python
insights = client.get_ad_insights(
    level="ad",
    fields=["spend", "actions", "impressions"],
    filtering=[{"field": "adset.id", "operator": "IN", "value": ["123"]}],
    time_range={"since": "2024-01-01", "until": "2024-01-31"}
)
```

##### `create_ad(adset_id: str, creative_id: str, name: str, status: str = "ACTIVE") -> str`
Create a new ad in Meta.

**Parameters:**
- `adset_id` (str): Ad set ID
- `creative_id` (str): Creative ID
- `name` (str): Ad name
- `status` (str): Ad status

**Returns:** str: Created ad ID

**Example:**
```python
ad_id = client.create_ad(
    adset_id="120231838265460160",
    creative_id="123456789",
    name="[TEST] My Ad"
)
```

##### `update_adset_budget(adset_id: str, budget: float) -> bool`
Update ad set budget.

**Parameters:**
- `adset_id` (str): Ad set ID
- `budget` (float): New budget amount

**Returns:** bool: Success status

**Example:**
```python
success = client.update_adset_budget("120231838265460160", 100.0)
```

##### `pause_ad(ad_id: str) -> bool`
Pause an ad.

**Parameters:**
- `ad_id` (str): Ad ID to pause

**Returns:** bool: Success status

**Example:**
```python
success = client.pause_ad("123456789")
```

### rules.py

Implements business logic and decision rules for automation.

#### Classes

##### `RuleEngine`
Main rule evaluation engine.

**Parameters:**
- `rules_config` (Dict[str, Any]): Rules configuration

**Example:**
```python
engine = RuleEngine(rules_config)
```

#### Methods

##### `evaluate_kill_rules(metrics: Metrics, rules: List[Dict]) -> Tuple[bool, str]`
Determine if ads should be killed based on rules.

**Parameters:**
- `metrics` (Metrics): Performance metrics
- `rules` (List[Dict]): Kill rules to evaluate

**Returns:** Tuple[bool, str]: (should_kill, reason)

**Example:**
```python
should_kill, reason = engine.evaluate_kill_rules(metrics, kill_rules)
```

##### `evaluate_advance_rules(metrics: Metrics, rules: List[Dict]) -> Tuple[bool, str]`
Determine if ads should advance to next stage.

**Parameters:**
- `metrics` (Metrics): Performance metrics
- `rules` (List[Dict]): Advance rules to evaluate

**Returns:** Tuple[bool, str]: (should_advance, reason)

**Example:**
```python
should_advance, reason = engine.evaluate_advance_rules(metrics, advance_rules)
```

##### `calculate_metrics(row: Dict[str, Any]) -> Metrics`
Calculate performance metrics from insight data.

**Parameters:**
- `row` (Dict[str, Any]): Insight data row

**Returns:** Metrics: Calculated metrics

**Example:**
```python
metrics = engine.calculate_metrics(insight_row)
```

### storage.py

Manages SQLite database for state persistence and logging.

#### Classes

##### `Store`
Main storage interface.

**Parameters:**
- `db_path` (str): Path to SQLite database

**Example:**
```python
store = Store("data/state.sqlite")
```

#### Methods

##### `log(entity_type: str, entity_id: str, action: str, level: str, stage: str, reason: str, meta: Dict[str, Any] = None) -> None`
Record system events in database.

**Parameters:**
- `entity_type` (str): Type of entity (e.g., "ad", "adset")
- `entity_id` (str): Entity identifier
- `action` (str): Action performed
- `level` (str): Log level ("info", "warn", "error")
- `stage` (str): Automation stage
- `reason` (str): Reason for action
- `meta` (Dict[str, Any], optional): Additional metadata

**Returns:** None

**Example:**
```python
store.log(
    entity_type="ad",
    entity_id="123456789",
    action="KILL",
    level="info",
    stage="testing",
    reason="CPA too high",
    meta={"cpa": 45.0, "threshold": 36.0}
)
```

##### `get_state(key: str) -> Any`
Retrieve stored state value.

**Parameters:**
- `key` (str): State key

**Returns:** Any: Stored value

**Example:**
```python
value = store.get_state("last_run_time")
```

##### `set_state(key: str, value: Any) -> None`
Store state value.

**Parameters:**
- `key` (str): State key
- `value` (Any): Value to store

**Returns:** None

**Example:**
```python
store.set_state("last_run_time", "2024-01-01T12:00:00Z")
```

##### `incr(key: str, amount: int = 1) -> int`
Increment counter value.

**Parameters:**
- `key` (str): Counter key
- `amount` (int): Amount to increment

**Returns:** int: New counter value

**Example:**
```python
count = store.incr("ads_created_today")
```

## Stage Modules

### testing.py

Handles the testing stage of new creative assets.

#### Functions

##### `run_testing_tick(client: MetaClient, settings: Dict[str, Any], engine: RuleEngine, store: Store, queue_df: pd.DataFrame, set_supabase_status: Callable, placements: List[str] = None, instagram_actor_id: str = None) -> Dict[str, Any]`
Main testing stage execution.

**Parameters:**
- `client` (MetaClient): Meta API client
- `settings` (Dict[str, Any]): Settings configuration
- `engine` (RuleEngine): Rule engine
- `store` (Store): Database store
- `queue_df` (pd.DataFrame): Creative queue
- `set_supabase_status` (Callable): Supabase status setter
- `placements` (List[str], optional): Ad placements
- `instagram_actor_id` (str, optional): Instagram actor ID

**Returns:** Dict[str, Any]: Testing results summary

**Example:**
```python
results = run_testing_tick(
    client=client,
    settings=settings,
    engine=engine,
    store=store,
    queue_df=queue_df,
    set_supabase_status=set_supabase_status
)
```

##### `launch_new_ads(client: MetaClient, settings: Dict[str, Any], store: Store, queue_df: pd.DataFrame, set_supabase_status: Callable) -> List[str]`
Launch new ads from creative queue.

**Parameters:**
- `client` (MetaClient): Meta API client
- `settings` (Dict[str, Any]): Settings configuration
- `store` (Store): Database store
- `queue_df` (pd.DataFrame): Creative queue
- `set_supabase_status` (Callable): Supabase status setter

**Returns:** List[str]: List of launched ad IDs

**Example:**
```python
launched_ads = launch_new_ads(client, settings, store, queue_df, set_supabase_status)
```

##### `evaluate_testing_performance(client: MetaClient, settings: Dict[str, Any], engine: RuleEngine, store: Store) -> Dict[str, Any]`
Evaluate ad performance and make decisions.

**Parameters:**
- `client` (MetaClient): Meta API client
- `settings` (Dict[str, Any]): Settings configuration
- `engine` (RuleEngine): Rule engine
- `store` (Store): Database store

**Returns:** Dict[str, Any]: Performance evaluation results

**Example:**
```python
results = evaluate_testing_performance(client, settings, engine, store)
```

### validation.py

Manages the validation stage with extended testing.

#### Functions

##### `run_validation_tick(client: MetaClient, settings: Dict[str, Any], engine: RuleEngine, store: Store) -> Dict[str, Any]`
Main validation stage execution.

**Parameters:**
- `client` (MetaClient): Meta API client
- `settings` (Dict[str, Any]): Settings configuration
- `engine` (RuleEngine): Rule engine
- `store` (Store): Database store

**Returns:** Dict[str, Any]: Validation results summary

**Example:**
```python
results = run_validation_tick(client, settings, engine, store)
```

##### `promote_to_scaling(client: MetaClient, settings: Dict[str, Any], store: Store, ad_ids: List[str]) -> List[str]`
Promote validated ads to scaling stage.

**Parameters:**
- `client` (MetaClient): Meta API client
- `settings` (Dict[str, Any]): Settings configuration
- `store` (Store): Database store
- `ad_ids` (List[str]): Ad IDs to promote

**Returns:** List[str]: Promoted ad IDs

**Example:**
```python
promoted_ads = promote_to_scaling(client, settings, store, ad_ids)
```

### scaling.py

Handles advanced scaling with portfolio management.

#### Functions

##### `run_scaling_tick(client: MetaClient, settings: Dict[str, Any], store: Store) -> Dict[str, Any]`
Main scaling stage execution.

**Parameters:**
- `client` (MetaClient): Meta API client
- `settings` (Dict[str, Any]): Settings configuration
- `store` (Store): Database store

**Returns:** Dict[str, Any]: Scaling results summary

**Example:**
```python
results = run_scaling_tick(client, settings, store)
```

##### `scale_budgets(client: MetaClient, settings: Dict[str, Any], store: Store, ad_ids: List[str]) -> Dict[str, Any]`
Scale budgets for winning ads.

**Parameters:**
- `client` (MetaClient): Meta API client
- `settings` (Dict[str, Any]): Settings configuration
- `store` (Store): Database store
- `ad_ids` (List[str]): Ad IDs to scale

**Returns:** Dict[str, Any]: Scaling results

**Example:**
```python
results = scale_budgets(client, settings, store, ad_ids)
```

##### `duplicate_creatives(client: MetaClient, settings: Dict[str, Any], store: Store, ad_ids: List[str]) -> List[str]`
Duplicate high-performing creatives.

**Parameters:**
- `client` (MetaClient): Meta API client
- `settings` (Dict[str, Any]): Settings configuration
- `store` (Store): Database store
- `ad_ids` (List[str]): Ad IDs to duplicate

**Returns:** List[str]: Duplicated ad IDs

**Example:**
```python
duplicated_ads = duplicate_creatives(client, settings, store, ad_ids)
```

## Utility Modules

### utils.py

Utility functions for common operations.

#### Functions

##### `now_local(tz_name: str = None) -> datetime`
Get current time in specified timezone.

**Parameters:**
- `tz_name` (str, optional): Timezone name

**Returns:** datetime: Current time

**Example:**
```python
now = now_local("Europe/Amsterdam")
```

##### `getenv_f(name: str, default: float) -> float`
Get float environment variable with default.

**Parameters:**
- `name` (str): Environment variable name
- `default` (float): Default value

**Returns:** float: Environment variable value or default

**Example:**
```python
budget = getenv_f("DAILY_BUDGET", 50.0)
```

##### `getenv_i(name: str, default: int) -> int`
Get integer environment variable with default.

**Parameters:**
- `name` (str): Environment variable name
- `default` (int): Default value

**Returns:** int: Environment variable value or default

**Example:**
```python
max_ads = getenv_i("MAX_ADS", 4)
```

##### `getenv_b(name: str, default: bool) -> bool`
Get boolean environment variable with default.

**Parameters:**
- `name` (str): Environment variable name
- `default` (bool): Default value

**Returns:** bool: Environment variable value or default

**Example:**
```python
dry_run = getenv_b("DRY_RUN", False)
```

##### `cfg(settings: Dict[str, Any], path: str, default: Any = None) -> Any`
Get configuration value using dot notation path.

**Parameters:**
- `settings` (Dict[str, Any]): Settings dictionary
- `path` (str): Dot notation path (e.g., "testing.daily_budget_eur")
- `default` (Any, optional): Default value

**Returns:** Any: Configuration value

**Example:**
```python
budget = cfg(settings, "testing.daily_budget_eur", 50.0)
```

##### `safe_f(value: Any, default: float = 0.0) -> float`
Safely convert value to float.

**Parameters:**
- `value` (Any): Value to convert
- `default` (float): Default value if conversion fails

**Returns:** float: Converted value or default

**Example:**
```python
spend = safe_f(row.get("spend"), 0.0)
```

##### `prettify_ad_name(name: str) -> str`
Clean and format ad name for display.

**Parameters:**
- `name` (str): Raw ad name

**Returns:** str: Formatted ad name

**Example:**
```python
clean_name = prettify_ad_name("[TEST] My Ad Name")
```

### slack.py

Slack notification and messaging functions.

#### Functions

##### `notify(message: str, level: str = "info") -> None`
Send notification to Slack.

**Parameters:**
- `message` (str): Message to send
- `level` (str): Message level ("info", "warn", "error")

**Returns:** None

**Example:**
```python
notify("Ad launched successfully", "info")
```

##### `alert_kill(ad_name: str, reason: str, cpa: float, roas: float) -> None`
Send kill alert to Slack.

**Parameters:**
- `ad_name` (str): Ad name
- `reason` (str): Kill reason
- `cpa` (float): CPA value
- `roas` (float): ROAS value

**Returns:** None

**Example:**
```python
alert_kill("My Ad", "CPA too high", 45.0, 1.2)
```

##### `alert_promote(ad_name: str, from_stage: str, to_stage: str, reason: str, budget: float) -> None`
Send promotion alert to Slack.

**Parameters:**
- `ad_name` (str): Ad name
- `from_stage` (str): Source stage
- `to_stage` (str): Target stage
- `reason` (str): Promotion reason
- `budget` (float): New budget

**Returns:** None

**Example:**
```python
alert_promote("My Ad", "testing", "validation", "Good performance", 100.0)
```

## Data Structures

### Metrics

Performance metrics for ad evaluation.

**Attributes:**
- `spend` (float): Total spend
- `impressions` (int): Total impressions
- `clicks` (int): Total clicks
- `purchases` (int): Total purchases
- `cpa` (float): Cost per acquisition
- `roas` (float): Return on ad spend
- `ctr` (float): Click-through rate

**Example:**
```python
metrics = Metrics(
    spend=100.0,
    impressions=10000,
    clicks=100,
    purchases=2,
    cpa=50.0,
    roas=2.0,
    ctr=0.01
)
```

### LogEntry

Database log entry structure.

**Attributes:**
- `entity_type` (str): Entity type
- `entity_id` (str): Entity ID
- `action` (str): Action performed
- `level` (str): Log level
- `stage` (str): Automation stage
- `reason` (str): Action reason
- `meta` (Dict[str, Any]): Additional metadata
- `created_at` (datetime): Creation timestamp

**Example:**
```python
entry = LogEntry(
    entity_type="ad",
    entity_id="123456789",
    action="KILL",
    level="info",
    stage="testing",
    reason="CPA too high",
    meta={"cpa": 45.0}
)
```

## Error Handling

### Common Exceptions

#### `MetaAPIError`
Raised when Meta API calls fail.

**Attributes:**
- `message` (str): Error message
- `code` (int): Error code
- `response` (Dict): API response

**Example:**
```python
try:
    client.create_ad(adset_id, creative_id, name)
except MetaAPIError as e:
    print(f"API Error: {e.message}")
```

#### `ConfigurationError`
Raised when configuration is invalid.

**Attributes:**
- `message` (str): Error message
- `field` (str): Configuration field

**Example:**
```python
try:
    validate_config(settings)
except ConfigurationError as e:
    print(f"Config Error: {e.message}")
```

## Type Hints

All functions include comprehensive type hints for better IDE support and documentation.

**Example:**
```python
def get_ad_insights(
    self,
    level: str,
    fields: List[str],
    filtering: Optional[List[Dict[str, Any]]] = None,
    time_range: Optional[Dict[str, str]] = None,
    action_attribution_windows: Optional[List[str]] = None,
    paginate: bool = True
) -> List[Dict[str, Any]]:
    """Retrieve ad performance insights from Meta API."""
    pass
```

## Examples

### Basic Usage

```python
from src.main import main
from src.meta_client import MetaClient, AccountAuth, ClientConfig
from src.storage import Store
from src.rules import RuleEngine

# Initialize components
store = Store("data/state.sqlite")
auth = AccountAuth(
    account_id="act_123456789",
    access_token="your_token",
    app_id="your_app_id",
    app_secret="your_secret"
)
client = MetaClient([auth], ClientConfig(), store=store)
engine = RuleEngine(rules_config)

# Run automation
main()
```

### Custom Integration

```python
# Custom testing stage
from src.stages.testing import run_testing_tick

results = run_testing_tick(
    client=client,
    settings=settings,
    engine=engine,
    store=store,
    queue_df=queue_df,
    set_supabase_status=set_supabase_status
)

print(f"Testing results: {results}")
```

### Direct API Usage

```python
# Direct Meta API usage
insights = client.get_ad_insights(
    level="ad",
    fields=["spend", "actions"],
    filtering=[{"field": "adset.id", "operator": "IN", "value": ["123"]}]
)

for insight in insights:
    print(f"Ad {insight['ad_id']}: ${insight['spend']}")
```
