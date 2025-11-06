# meta_client.py
from __future__ import annotations

import hashlib
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict, deque
from threading import Lock
import json

import logging
import requests

# Import moved to avoid circular dependency - will import locally when needed
from .slack import notify

logger = logging.getLogger(__name__)
from infrastructure.error_handling import (
    retry_with_backoff,
    enhanced_retry_with_backoff,
    with_circuit_breaker,
    circuit_breaker_manager,
)

# Import rate limit manager
try:
    from infrastructure.rate_limit_manager import get_rate_limit_manager, RateLimitType
    RATE_LIMIT_MANAGER_AVAILABLE = True
except ImportError:
    RATE_LIMIT_MANAGER_AVAILABLE = False

USE_SDK = False
try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.adobjects.adset import AdSet
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.ad import Ad
    from facebook_business.adobjects.campaign import Campaign
    from facebook_business.exceptions import FacebookRequestError
    USE_SDK = True
except Exception:  # pragma: no cover
    USE_SDK = False


# -------------------------
# Environment & guards
# -------------------------
META_RETRY_MAX          = int(os.getenv("META_RETRY_MAX", "6") or 6)
META_BACKOFF_BASE       = float(os.getenv("META_BACKOFF_BASE", "1.0") or 1.0)
META_TIMEOUT            = float(os.getenv("META_TIMEOUT", "30") or 30)
META_WRITE_COOLDOWN_SEC = int(os.getenv("META_WRITE_COOLDOWN_SEC", "10") or 10)

BUDGET_MIN          = float(os.getenv("BUDGET_MIN", "5") or 5.0)
BUDGET_MAX          = float(os.getenv("BUDGET_MAX", "50000") or 50000.0)
BUDGET_MAX_STEP_PCT = float(os.getenv("BUDGET_MAX_STEP_PCT", "200") or 200.0)

# Circuit breaker
CB_FAILS     = int(os.getenv("META_CB_FAILS", "5") or 5)
CB_RESET_SEC = int(os.getenv("META_CB_RESET_SEC", "120") or 120)

# Rate limiting - Enhanced for UI compatibility
META_REQUEST_DELAY = float(os.getenv("META_REQUEST_DELAY", "0.8") or 0.8)  # Increased delay
META_REQUEST_JITTER = 0.3  # Random jitter to avoid burst alignment
META_MAX_CONCURRENT_INSIGHTS = int(os.getenv("META_MAX_CONCURRENT_INSIGHTS", "3") or 3)  # Hard concurrency cap
META_USAGE_THRESHOLD = 0.8  # Pause when usage > 80%

# Enhanced rate limiting configuration
META_API_TIER = os.getenv("META_API_TIER", "development")  # "development" or "standard"
META_MAX_SCORE_DEV = 60  # Development tier max score
META_MAX_SCORE_STANDARD = 9000  # Standard tier max score
META_SCORE_DECAY_SEC = 300  # Score decay rate in seconds
META_BLOCK_DURATION_DEV = 300  # Block duration for dev tier
META_BLOCK_DURATION_STANDARD = 60  # Block duration for standard tier

# Business Use Case (BUC) rate limits
META_BUC_ENABLED = os.getenv("META_BUC_ENABLED", "true").lower() == "true"
META_BUC_HEADERS = {
    "ads_management": "X-Business-Use-Case: ads_management",
    "custom_audience": "X-Business-Use-Case: custom_audience", 
    "ads_insights": "X-Business-Use-Case: ads_insights",
    "catalog_management": "X-Business-Use-Case: catalog_management",
    "catalog_batch": "X-Business-Use-Case: catalog_batch"
}

# Naming & compliance
CAMPAIGN_NAME_RE = re.compile(r"^\[(TEST|VALID|SCALE|SCALE-CBO|ASC\+)\]\s+Brava\s+-\s+(ABO|CBO)\s+-\s+US Men$")
ADSET_NAME_RE    = re.compile(r"^\[(TEST|VALID|SCALE|ASC\+)\]\s+.+$")
AD_NAME_RE       = re.compile(r"^\[(TEST|VALID|SCALE|ASC\+)\]\s+.+$")
FORBIDDEN_TERMS  = tuple(x.strip().lower() for x in os.getenv("FORBIDDEN_TERMS", "cures,miracle,guaranteed").split(","))

HUMAN_CONFIRM_JUMP_PCT = float(os.getenv("HUMAN_CONFIRM_JUMP_PCT", "200") or 200.0)

# Account metadata (new: Amsterdam/EUR)
ACCOUNT_TIMEZONE  = os.getenv("ACCOUNT_TZ") or os.getenv("ACCOUNT_TIMEZONE") or "Europe/Amsterdam"
ACCOUNT_CURRENCY  = os.getenv("ACCOUNT_CURRENCY", "EUR")
ACCOUNT_CCY_SYM   = os.getenv("ACCOUNT_CURRENCY_SYMBOL", "€")


# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class AccountAuth:
    account_id: str                # can be "act_123" or "123"
    access_token: str
    app_id: str
    app_secret: str
    api_version: Optional[str] = None  # defaulted below


@dataclass
class ClientConfig:
    # changed default timezone to Europe/Amsterdam
    timezone: str = ACCOUNT_TIMEZONE
    attribution_click_days: int = 7
    attribution_view_days: int = 1
    roas_source: str = "computed"
    # added currency for clarity
    currency: str = ACCOUNT_CURRENCY
    currency_symbol: str = ACCOUNT_CCY_SYM

    fields_default: Tuple[str, ...] = (
        "ad_id", "ad_name", "adset_id", "campaign_id",
        "spend", "impressions", "clicks", "reach", "unique_clicks",
        "actions", "action_values", "purchase_roas",
    )
    breakdowns_default: Tuple[str, ...] = ()
    stage_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # feature flags
    enable_creative_uploads: bool = True
    enable_duplication: bool = True
    enable_budget_updates: bool = True
    require_name_compliance: bool = True

    # safety & pacing
    write_cooldown_sec: int = META_WRITE_COOLDOWN_SEC
    budget_min: float = BUDGET_MIN
    budget_max: float = BUDGET_MAX
    budget_step_cap_pct: float = BUDGET_MAX_STEP_PCT


# -------------------------
# Helpers
# -------------------------
def _hash_idempotency(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def _contains_forbidden(texts: Iterable[str]) -> Optional[str]:
    for t in texts:
        s = (t or "").lower()
        for bad in FORBIDDEN_TERMS:
            if bad and bad in s:
                return bad
    return None


def _s(x: Any) -> str:
    if x is None:
        return ""
    if callable(x):
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if callable(obj):
        return None
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if callable(k):
                continue
            out[_s(k)] = _sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return _s(obj)


def _clean_story_link(link_url: Optional[str], utm_params: Optional[str]) -> str:
    base = _s(link_url or os.getenv("STORE_URL") or "https://example.com")
    u = _s(utm_params).lstrip("?")
    if not u:
        return base
    sep = "&" if ("?" in base) else "?"
    return f"{base}{sep}{u}"


def _normalize_account_id(account_id: str) -> Tuple[str, str]:
    """Returns (numeric_id, act_prefixed_id). Accepts either '123' or 'act_123'."""
    aid = (account_id or "").strip()
    num = aid[4:] if aid.startswith("act_") else aid
    return num, f"act_{num}"


def _is_digits(s: Optional[str]) -> bool:
    return bool(s) and str(s).strip().isdigit()


def _maybe_call(v: Any) -> Any:
    try:
        return v() if callable(v) else v
    except Exception:
        return None


# -------------------------
# Rate Limiting Manager
# -------------------------
class RateLimitManager:
    """
    Comprehensive rate limiting manager for Meta Marketing API.
    Handles all rate limiting types: API-level, Business Use Case, Insights Platform, etc.
    """
    
    def __init__(self, api_tier: str = "development", store: Optional[Any] = None):
        self.api_tier = api_tier.lower()
        self.store = store
        self.lock = Lock()
        
        # API-level scoring (reads=1pt, writes=3pts)
        self.current_score = 0
        self.max_score = META_MAX_SCORE_DEV if self.api_tier == "development" else META_MAX_SCORE_STANDARD
        self.block_duration = META_BLOCK_DURATION_DEV if self.api_tier == "development" else META_BLOCK_DURATION_STANDARD
        self.blocked_until = 0
        
        # Score tracking with decay
        self.score_history = deque(maxlen=1000)  # Keep last 1000 requests
        
        # Business Use Case tracking
        self.buc_usage = defaultdict(int)  # BUC -> usage count
        self.buc_reset_times = defaultdict(float)  # BUC -> next reset time
        
        # Ad account level limits
        self.adset_budget_changes = defaultdict(list)  # adset_id -> list of change timestamps
        self.account_spend_changes = []  # List of spend change timestamps
        
        # App-level limits (for Insights Platform)
        self.app_level_blocked_until = 0
        
        # Usage monitoring for UI compatibility
        self.insights_throttle_usage = 0.0
        self.buc_usage_percent = defaultdict(float)  # BUC -> usage percentage
        self.ad_account_usage_percent = defaultdict(float)  # Ad account -> usage percentage
        self.app_usage_percent = 0.0
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.last_error_reset = time.time()
        
    def _cleanup_old_scores(self):
        """Remove scores older than decay period."""
        cutoff = time.time() - META_SCORE_DECAY_SEC
        while self.score_history and self.score_history[0][0] < cutoff:
            old_score = self.score_history.popleft()[1]
            self.current_score = max(0, self.current_score - old_score)
    
    def _cleanup_buc_usage(self):
        """Reset BUC usage counters when reset time is reached."""
        current_time = time.time()
        for buc, reset_time in list(self.buc_reset_times.items()):
            if current_time >= reset_time:
                self.buc_usage[buc] = 0
                self.buc_reset_times[buc] = current_time + 3600  # Reset every hour
    
    def can_make_request(self, request_type: str = "read", buc_type: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if a request can be made based on all rate limiting rules.
        Returns (can_make, reason_if_blocked)
        """
        with self.lock:
            current_time = time.time()
            
            # Clean up old data
            self._cleanup_old_scores()
            self._cleanup_buc_usage()
            
            # Check if currently blocked
            if current_time < self.blocked_until:
                return False, f"API-level blocked until {self.blocked_until - current_time:.1f}s"
            
            if current_time < self.app_level_blocked_until:
                return False, f"App-level blocked until {self.app_level_blocked_until - current_time:.1f}s"
            
            # Check API-level scoring
            request_score = 3 if request_type == "write" else 1
            if self.current_score + request_score > self.max_score:
                return False, f"API score limit exceeded ({self.current_score}/{self.max_score})"
            
            # Check BUC limits if applicable
            if buc_type and META_BUC_ENABLED:
                buc_limit = self._get_buc_limit(buc_type)
                if self.buc_usage[buc_type] >= buc_limit:
                    reset_in = self.buc_reset_times[buc_type] - current_time
                    return False, f"BUC {buc_type} limit exceeded ({self.buc_usage[buc_type]}/{buc_limit}), resets in {reset_in:.1f}s"
            
            return True, ""
    
    def record_request(self, request_type: str = "read", buc_type: Optional[str] = None, endpoint: str = ""):
        """Record a request and update all relevant counters."""
        with self.lock:
            current_time = time.time()
            
            # Update API-level scoring
            request_score = 3 if request_type == "write" else 1
            self.score_history.append((current_time, request_score))
            self.current_score += request_score
            
            # Update BUC usage
            if buc_type and META_BUC_ENABLED:
                self.buc_usage[buc_type] += 1
                if buc_type not in self.buc_reset_times:
                    self.buc_reset_times[buc_type] = current_time + 3600  # Reset every hour
            
            # Log to store if available
            if self.store:
                try:
                    self.store.log("rate_limit", "request_recorded", {
                        "type": request_type,
                        "buc_type": buc_type,
                        "endpoint": endpoint,
                        "score": request_score,
                        "current_score": self.current_score,
                        "max_score": self.max_score
                    }, level="debug", stage="RATE_LIMIT")
                except Exception:
                    pass
    
    def handle_rate_limit_error(self, error_data: Dict[str, Any], endpoint: str) -> float:
        """
        Handle rate limit errors and return wait time.
        Returns the number of seconds to wait before retry.
        """
        with self.lock:
            code = error_data.get("error", {}).get("code")
            subcode = error_data.get("error", {}).get("error_subcode")
            message = error_data.get("error", {}).get("message", "")
            current_time = time.time()
            
            # Reset error counts every hour
            if current_time - self.last_error_reset > 3600:
                self.error_counts.clear()
                self.last_error_reset = current_time
            
            self.error_counts[f"{code}_{subcode}"] += 1
            
            # Handle different rate limit error types with enhanced backoff
            if code == 4:  # Application request limit reached
                if subcode == 1504022:  # Ads Insights Platform rate limit (UI interference)
                    wait_time = 120.0  # Increased from 60s to 120s
                    self.app_level_blocked_until = current_time + wait_time
                    notify(f"⏳ Ads Insights Platform rate limit hit (UI interference). Blocked for {wait_time:.1f}s")
                    return wait_time
                elif subcode == 1504039:  # General app-level rate limit
                    wait_time = 90.0  # Increased from 30s to 90s
                    self.app_level_blocked_until = current_time + wait_time
                    notify(f"⏳ App-level rate limit hit. Blocked for {wait_time:.1f}s")
                    return wait_time
                else:  # General application rate limit
                    wait_time = min(60.0, META_BACKOFF_BASE * 8)
                    self.blocked_until = current_time + wait_time
                    notify(f"⏳ Application rate limit hit. Blocked for {wait_time:.1f}s")
                    return wait_time
            
            elif code == 17:  # User request limit reached
                if subcode == 2446079:  # Ad account level API limit
                    wait_time = self.block_duration
                    self.blocked_until = current_time + wait_time
                    notify(f"⏳ Ad account API limit reached. Blocked for {wait_time:.1f}s")
                    return wait_time
                elif subcode == 1885172:  # Spend limit changes (10/day)
                    wait_time = 3600.0  # Wait 1 hour
                    notify(f"⏳ Spend limit change limit reached (10/day). Wait 1 hour.")
                    return wait_time
            
            elif code == 613:  # Ad account level limits
                if subcode == 1487742:  # Too many calls from ad account
                    wait_time = 60.0
                    self.blocked_until = current_time + wait_time
                    notify(f"⏳ Too many calls from ad account. Blocked for {wait_time:.1f}s")
                    return wait_time
                elif subcode == 1487632:  # Ad set budget change limit (4/hour)
                    wait_time = 3600.0  # Wait 1 hour
                    notify(f"⏳ Ad set budget change limit reached (4/hour). Wait 1 hour.")
                    return wait_time
                elif subcode == 1487225:  # Ad creation limit
                    wait_time = 60.0
                    notify(f"⏳ Ad creation limit reached. Wait {wait_time:.1f}s")
                    return wait_time
                else:  # General ad account limit
                    wait_time = 60.0
                    self.blocked_until = current_time + wait_time
                    notify(f"⏳ Ad account limit reached. Blocked for {wait_time:.1f}s")
                    return wait_time
            
            elif code in (80000, 80003, 80004, 80014):  # Business Use Case rate limits
                wait_time = 60.0
                notify(f"⏳ Business Use Case rate limit hit. Wait {wait_time:.1f}s")
                return wait_time
            
            # Default handling for rate limit related errors
            if "rate" in message.lower() or "limit" in message.lower():
                wait_time = min(30.0, META_BACKOFF_BASE * 4)
                notify(f"⏳ Rate limit error detected. Wait {wait_time:.1f}s")
                return wait_time
            
            # No specific rate limit detected
            return 0.0
    
    def _get_buc_limit(self, buc_type: str) -> int:
        """Get the rate limit for a specific Business Use Case."""
        if self.api_tier == "development":
            limits = {
                "ads_management": 300,
                "custom_audience": 5000,
                "ads_insights": 600,
                "catalog_management": 20000,
                "catalog_batch": 200
            }
        else:  # standard tier
            limits = {
                "ads_management": 100000,
                "custom_audience": 190000,
                "ads_insights": 190000,
                "catalog_management": 20000,
                "catalog_batch": 200
            }
        
        return limits.get(buc_type, 1000)  # Default limit
    
    def can_change_budget(self, adset_id: str) -> bool:
        """Check if ad set budget can be changed (4 times per hour limit)."""
        with self.lock:
            current_time = time.time()
            cutoff = current_time - 3600  # 1 hour ago
            
            # Clean old changes
            self.adset_budget_changes[adset_id] = [
                ts for ts in self.adset_budget_changes[adset_id] if ts > cutoff
            ]
            
            return len(self.adset_budget_changes[adset_id]) < 4
    
    def record_budget_change(self, adset_id: str):
        """Record an ad set budget change."""
        with self.lock:
            self.adset_budget_changes[adset_id].append(time.time())
    
    def can_change_spend_limit(self) -> bool:
        """Check if account spend limit can be changed (10 times per day limit)."""
        with self.lock:
            current_time = time.time()
            cutoff = current_time - 86400  # 24 hours ago
            
            # Clean old changes
            self.account_spend_changes = [
                ts for ts in self.account_spend_changes if ts > cutoff
            ]
            
            return len(self.account_spend_changes) < 10
    
    def record_spend_limit_change(self):
        """Record an account spend limit change."""
        with self.lock:
            self.account_spend_changes.append(time.time())
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiting status."""
        with self.lock:
            current_time = time.time()
            self._cleanup_old_scores()
            self._cleanup_buc_usage()
            
            return {
                "api_tier": self.api_tier,
                "current_score": self.current_score,
                "max_score": self.max_score,
                "score_usage_pct": (self.current_score / self.max_score) * 100,
                "blocked_until": self.blocked_until,
                "app_blocked_until": self.app_level_blocked_until,
                "buc_usage": dict(self.buc_usage),
                "error_counts": dict(self.error_counts),
                "recent_requests": len(self.score_history)
            }


# -------------------------
# MetaClient
# -------------------------
class MetaClient:
    """
    Reads can use SDK. Writes (creatives/ads) are HTTP-first to avoid flaky SDK paths.

    Budgets in this client are expressed in the **ad account currency** (now EUR),
    and sent to Graph as integer "cents" of that currency, as Meta expects.
    """

    def __init__(
        self,
        accounts: Union[List[AccountAuth], AccountAuth],
        cfg: Optional[ClientConfig] = None,
        *,
        dry_run: bool = True,
        store: Optional[Any] = None,
        tenant_id: Optional[str] = None,
    ):
        self.cfg = cfg or ClientConfig()
        self.store = store
        self.dry_run = bool(dry_run)
        self.tenant_id = tenant_id or "default"

        self.accounts: List[AccountAuth] = accounts if isinstance(accounts, list) else [accounts]
        if not self.accounts:
            raise ValueError("At least one AccountAuth is required.")

        # default API version
        for acc in self.accounts:
            if not acc.api_version:
                acc.api_version = "v23.0"

        # normalize & cache account ids per index
        self._acct_num: Dict[int, str] = {}
        self._acct_act: Dict[int, str] = {}
        for i, acc in enumerate(self.accounts):
            num, act = _normalize_account_id(acc.account_id)
            self._acct_num[i] = num
            self._acct_act[i] = act

        self._active_idx = 0
        self._sdk_inited_for: Optional[str] = None
        self._fail_count: Dict[str, int] = {}
        self._cb_open_until: Dict[str, float] = {}
        self._last_write_ts = 0.0

        # Initialize rate limiting manager
        self.rate_limit_manager = RateLimitManager(
            api_tier=META_API_TIER,
            store=self.store
        )

        self._init_sdk_if_needed()

    # ------------- SDK init / failover -------------
    @property
    def account(self) -> AccountAuth:
        return self.accounts[self._active_idx]

    @property
    def ad_account_id_numeric(self) -> str:
        return self._acct_num[self._active_idx]

    @property
    def ad_account_id_act(self) -> str:
        return self._acct_act[self._active_idx]  # 'act_<id>'

    def _init_sdk_if_needed(self):
        if self.dry_run or not USE_SDK:
            return
        acct = self.account
        if self._sdk_inited_for == self.ad_account_id_act:
            return
        FacebookAdsApi.init(acct.app_id, acct.app_secret, acct.access_token, api_version=acct.api_version)
        self._sdk_inited_for = self.ad_account_id_act

    def _failover_account(self):
        if len(self.accounts) <= 1:
            return False
        self._active_idx = (self._active_idx + 1) % len(self.accounts)
        self._init_sdk_if_needed()
        return True

    # ------------- Circuit breaker / retry -------------
    def _cb_open(self, key: str) -> bool:
        until = self._cb_open_until.get(key)
        return bool(until and time.time() < until)

    def _cb_fail(self, key: str):
        cnt = self._fail_count.get(key, 0) + 1
        self._fail_count[key] = cnt
        if cnt >= CB_FAILS:
            self._cb_open_until[key] = time.time() + CB_RESET_SEC
            if self.store:
                try:
                    self.store.log("account", self.ad_account_id_act, "CB_OPEN", f"{key}", level="warn", stage="ACCOUNT")
                except Exception:
                    pass

    def _cb_success(self, key: str):
        self._fail_count[key] = 0
        self._cb_open_until.pop(key, None)

    def _retry(self, key: str, fn: Callable, *args, **kwargs):
        if self._cb_open(key):
            raise RuntimeError(f"Circuit open for {key}")
        last_exc = None
        for attempt in range(META_RETRY_MAX + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if self.dry_run:
                    raise
                retriable = True
                retry_after = None
                wait = META_BACKOFF_BASE * (2 ** attempt) * (0.75 + random.random() * 0.5)

                if USE_SDK and isinstance(e, FacebookRequestError):
                    code = _maybe_call(getattr(e, "api_error_code", None))
                    status = _maybe_call(getattr(e, "http_status", None))
                    headers = _maybe_call(getattr(e, "http_headers", None)) or {}
                    subcode = _maybe_call(getattr(e, "api_error_subcode", None))
                    try:
                        if "Retry-After" in headers:
                            retry_after = float(headers.get("Retry-After"))
                    except Exception:
                        retry_after = None
                    
                    # Handle rate limit errors specifically using the rate limit manager
                    if code in (4, 17, 613) or (code >= 80000 and code <= 80014):
                        error_data = {"error": {"code": code, "error_subcode": subcode, "message": str(e)}}
                        wait_time = self.rate_limit_manager.handle_rate_limit_error(error_data, key)
                        if wait_time > 0:
                            time.sleep(wait_time)
                        retriable = True
                    else:
                        retriable = (code in (4, 17, 613)) or (isinstance(status, int) and 500 <= status < 600)

                if retry_after:
                    time.sleep(max(0.5, retry_after))
                    continue
                if retriable and attempt < META_RETRY_MAX:
                    time.sleep(min(wait, 8.0))
                    continue
                last_exc = e
                break

        self._cb_fail(key)
        if self.store:
            try:
                self.store.log("account", self.ad_account_id_act, "META_API_ERROR", f"{key}", level="error", stage="ACCOUNT", reason=str(last_exc)[:300] if last_exc else key)
            except (AttributeError, TypeError, ValueError):
                # Store logging failed - non-critical
                pass
        raise last_exc if last_exc else RuntimeError("Meta API error")

    # ------------- Cooldown/pacing for writes -------------
    def _cooldown(self):
        delta = time.time() - self._last_write_ts
        need = max(0.0, self.cfg.write_cooldown_sec - delta)
        if need > 0:
            time.sleep(need)
        self._last_write_ts = time.time()

    # ------------- HTTP helpers -------------
    def _graph_url(self, endpoint: str) -> str:
        ver = self.account.api_version or "v23.0"
        return f"https://graph.facebook.com/{ver}/{self.ad_account_id_act}/{endpoint.lstrip('/')}"

    def _handle_rate_limit_error(self, error_data: Dict[str, Any], endpoint: str) -> None:
        """Handle Facebook API rate limit errors with appropriate retry logic."""
        wait_time = self.rate_limit_manager.handle_rate_limit_error(error_data, endpoint)
        if wait_time > 0:
            time.sleep(wait_time)

    def _request_delay(self) -> None:
        """Add delay between API requests to prevent rate limiting with jitter."""
        if META_REQUEST_DELAY > 0:
            import random
            jitter = random.uniform(-META_REQUEST_JITTER, META_REQUEST_JITTER)
            delay = max(0.1, META_REQUEST_DELAY + jitter)  # Minimum 0.1s
            time.sleep(delay)

    def _graph_post(self, endpoint: str, payload: Dict[str, Any], buc_type: str = "ads_management") -> Dict[str, Any]:
        # Check rate limits before making request
        can_make, reason = self.rate_limit_manager.can_make_request("write", buc_type)
        if not can_make:
            raise RuntimeError(f"Rate limit prevented request: {reason}")
        
        self._request_delay()  # Add delay to prevent rate limiting
        url = self._graph_url(endpoint)
        data = _sanitize(payload)
        data["access_token"] = self.account.access_token
        
        # Add BUC header if enabled
        headers = {}
        if META_BUC_ENABLED and buc_type in META_BUC_HEADERS:
            header_line = META_BUC_HEADERS[buc_type]
            if ":" in header_line:
                key, value = header_line.split(":", 1)
                headers[key.strip()] = value.strip()
        
        for attempt in range(META_RETRY_MAX + 1):
            try:
                r = requests.post(url, json=data, headers=headers, timeout=META_TIMEOUT)
                
                # Record the request
                self.rate_limit_manager.record_request("write", buc_type, endpoint)
                
                if r.status_code >= 400:
                    try:
                        err = r.json()
                    except Exception:
                        err = {"error": {"message": r.text}}
                    
                    # Check for rate limit errors
                    code = err.get("error", {}).get("code")
                    subcode = err.get("error", {}).get("error_subcode")
                    
                    # Handle rate limit errors
                    if code in (4, 17, 613) or (code >= 80000 and code <= 80014):
                        if attempt < META_RETRY_MAX:
                            self._handle_rate_limit_error(err, endpoint)
                            continue
                    
                    msg = f"Graph POST {endpoint} {r.status_code}: {err}"
                    try:
                        if code == 100 and subcode == 33:
                            msg += " - Hint: check ad account id, token scopes (ads_management), and account access."
                    except Exception:
                        pass
                    raise RuntimeError(msg)
                
                try:
                    return r.json()
                except Exception:
                    return {"ok": True, "text": r.text}
                    
            except Exception as e:
                if attempt < META_RETRY_MAX:
                    wait = META_BACKOFF_BASE * (2 ** attempt) * (0.75 + random.random() * 0.5)
                    time.sleep(min(wait, 30.0))
                    continue
                raise e

    def _graph_get_object(self, object_path: str, params: Optional[Dict[str, Any]] = None, buc_type: str = "ads_management") -> Dict[str, Any]:
        """GET for absolute objects like '/{id}' or '/{id}/thumbnails' (not under ad account path)."""
        # Check rate limits before making request
        can_make, reason = self.rate_limit_manager.can_make_request("read", buc_type)
        if not can_make:
            raise RuntimeError(f"Rate limit prevented request: {reason}")
        
        self._request_delay()  # Add delay to prevent rate limiting
        ver = self.account.api_version or "v23.0"
        url = f"https://graph.facebook.com/{ver}/{object_path.lstrip('/')}"
        qp = dict(params or {})
        qp["access_token"] = self.account.access_token
        
        # Add BUC header if enabled
        headers = {}
        if META_BUC_ENABLED and buc_type in META_BUC_HEADERS:
            header_line = META_BUC_HEADERS[buc_type]
            if ":" in header_line:
                key, value = header_line.split(":", 1)
                headers[key.strip()] = value.strip()
        
        for attempt in range(META_RETRY_MAX + 1):
            try:
                r = requests.get(url, params=qp, headers=headers, timeout=META_TIMEOUT)
                
                # Record the request
                self.rate_limit_manager.record_request("read", buc_type, object_path)
                
                if r.status_code >= 400:
                    try:
                        err = r.json()
                    except Exception:
                        err = {"error": {"message": r.text}}
                    
                    # Check for rate limit errors
                    code = err.get("error", {}).get("code")
                    subcode = err.get("error", {}).get("error_subcode")
                    
                    # Handle rate limit errors
                    if code in (4, 17, 613) or (code >= 80000 and code <= 80014):
                        if attempt < META_RETRY_MAX:
                            self._handle_rate_limit_error(err, object_path)
                            continue
                    
                    raise RuntimeError(f"Graph GET {object_path} {r.status_code}: {err}")
                
                return r.json()
                
            except Exception as e:
                if attempt < META_RETRY_MAX:
                    wait = META_BACKOFF_BASE * (2 ** attempt) * (0.75 + random.random() * 0.5)
                    time.sleep(min(wait, 30.0))
                    continue
                raise e

    # ------------- Compliance & idempotency -------------
    def _check_names(self, *, campaign: Optional[str] = None, adset: Optional[str] = None, ad: Optional[str] = None):
        if not self.cfg.require_name_compliance:
            return
        if campaign and not CAMPAIGN_NAME_RE.match(campaign):
            raise ValueError(f"Campaign name not compliant: {campaign}")
        if adset and not ADSET_NAME_RE.match(adset):
            raise ValueError(f"Ad set name not compliant: {adset}")
        if ad and not AD_NAME_RE.match(ad):
            raise ValueError(f"Ad name not compliant: {ad}")

    def _idempotent_guard(self, entity_id: str, op: str, params: Dict[str, Any], ttl_sec: int = 3600) -> bool:
        key = f"{self.tenant_id}:idem:{_hash_idempotency(entity_id, op, str(sorted(params.items())))}"
        if not self.store:
            return True
        try:
            if self.store.get_counter(key) > 0:
                return False
            self.store.set_counter(key, 1)
        except Exception:
            return True
        return True

    # ------------- Insights (READ) -------------
    def get_ad_insights(
        self,
        *,
        level: str = "ad",
        time_range: Optional[Dict[str, str]] = None,
        filtering: Optional[List[Any]] = None,
        limit: int = 500,
        fields: Optional[List[str]] = None,
        breakdowns: Optional[List[str]] = None,
        action_attribution_windows: Optional[List[str]] = None,
        paginate: bool = True,
        stage: Optional[str] = None,
        date_preset: Optional[str] = None,  # accepts presets; "lifetime" will be mapped to "maximum"
    ) -> List[Dict[str, Any]]:
        """
        Fetch insights. If `date_preset` is provided (e.g., "maximum", "today", "yesterday",
        "last_7d", "last_30d"), it is sent to the API and `time_range` is ignored.
        Some API versions do NOT accept "lifetime"; we auto-map "lifetime" -> "maximum".
        """
        # Normalize preset for API versions that don't accept "lifetime"
        normalized_preset = (date_preset or "").strip().lower() or None
        if normalized_preset == "lifetime":
            normalized_preset = "maximum"

        # Import locally to avoid circular dependency
        from infrastructure.utils import today_ymd, yesterday_ymd
        tr = None if normalized_preset else (time_range or {"since": yesterday_ymd(), "until": today_ymd()})

        if self.dry_run or not USE_SDK:
            # dev mock (values are in EUR conceptually, but numbers are examples)
            def mock_row(idx, stage_, spend, clicks, imps, purchases, revenue):
                return {
                    "ad_id": f"AD_{stage_}_{idx}",
                    "ad_name": f"[{stage_}] Creative_{idx}",
                    "adset_id": f"AS_{stage_}_1",
                    "campaign_id": f"CP_{stage_}_1",
                    "spend": spend,
                    "impressions": imps,
                    "clicks": clicks,
                    "reach": max(1, int(imps * 0.6)),
                    "unique_clicks": max(0, int(clicks * 0.8)),
                    "actions": [
                        {"action_type": "purchase", "value": str(purchases)},
                        {"action_type": "add_to_cart", "value": str(max(0, purchases * 2 - 1))},
                    ],
                    "action_values": [{"action_type": "purchase", "value": str(revenue)}],
                    "purchase_roas": [{"value": float(revenue) / spend if spend > 0 else 0.0}],
                }

            if normalized_preset in ("maximum",):
                rows = [
                    mock_row(1, "TEST", 65.00, 90, 9000, 0, 0.0),
                    mock_row(2, "TEST", 140.0, 160, 15000, 2, 160.0),
                    mock_row(3, "VALID", 200.0, 210, 19000, 3, 300.0),
                ]
            else:
                rows = [
                    mock_row(1, "TEST", 22.17, 35, 4200, 0, 0.0),
                    mock_row(2, "TEST", 38.76, 50, 5200, 1, 73.0),
                    mock_row(3, "VALID", 41.22, 48, 3900, 1, 90.0),
                    mock_row(4, "SCALE", 128.2, 120, 14000, 5, 520.0),
                ]
            if stage:
                rows = [r for r in rows if r["ad_name"].startswith(f"[{stage.upper()}]")]
            return rows

        self._init_sdk_if_needed()
        use_fields = fields or list(self.cfg.fields_default)
        params: Dict[str, Any] = {
            "level": level,
            "filtering": filtering or [],
            "limit": max(1, min(1000, int(limit))),
            "action_attribution_windows": action_attribution_windows
            or [f"{self.cfg.attribution_click_days}d_click", f"{self.cfg.attribution_view_days}d_view"],
        }
        if breakdowns:
            params["breakdowns"] = breakdowns

        # choose between time_range and date_preset
        if normalized_preset:
            params["date_preset"] = normalized_preset
        else:
            params["time_range"] = tr

        rows: List[Dict[str, Any]] = []

        def _get(after: Optional[str] = None):
            p = dict(params)
            if after:
                p["after"] = after
            return AdAccount(self.ad_account_id_act).get_insights(fields=use_fields, params=p)

        # primary attempt
        cursor = self._retry("insights", _get)
        rows.extend(list(cursor))

        if paginate:
            try:
                paging = cursor.get("paging") if hasattr(cursor, "get") else getattr(cursor, "paging", None)
                while paging and paging.get("cursors", {}).get("after"):
                    after = paging["cursors"]["after"]
                    cursor = self._retry("insights", _get, after)
                    rows.extend(list(cursor))
                    paging = cursor.get("paging") if hasattr(cursor, "get") else getattr(cursor, "paging", None)
            except Exception:
                pass

        return rows

    def list_ads_in_adset(self, adset_id: str) -> List[Dict[str, Any]]:
        if self.dry_run or not USE_SDK:
            return [{"id": f"{adset_id}_AD_{i}", "name": f"[TEST] Mock_{i}", "status": "ACTIVE"} for i in range(1, 5)]
        self._init_sdk_if_needed()
        ads = []
        def _fetch():
            return AdSet(adset_id).get_ads(fields=["id", "name", "status"])
        cursor = self._retry("list_ads", _fetch)
        for a in cursor:
            ads.append({"id": a["id"], "name": a.get("name"), "status": a.get("status")})
        return ads

    # ----- Helper: get current ad set budget (account currency, e.g., EUR) -----
    def get_adset_budget(self, adset_id: str) -> Optional[float]:
        """
        Returns the daily budget in account currency (EUR for this account).
        """
        if self.dry_run or not USE_SDK:
            return 100.0
        self._init_sdk_if_needed()
        def _fetch():
            return AdSet(adset_id).api_get(fields=["daily_budget"])
        try:
            res = self._retry("get_adset_budget", _fetch)
            cents = res.get("daily_budget")
            if cents is None:
                return None
            try:
                return float(int(cents)) / 100.0
            except Exception:
                return float(cents) / 100.0
        except Exception:
            return None

    # Back-compat alias (was USD; now account currency)
    def get_adset_budget_usd(self, adset_id: str) -> Optional[float]:
        return self.get_adset_budget(adset_id)

    # ----- Update budget (account currency, e.g., EUR) -----
    def update_adset_budget(
        self,
        adset_id: str,
        daily_budget: float,
        *,
        current_budget: Optional[float] = None,
        human_confirm: bool = False
    ):
        """
        Update daily budget in account currency (EUR). Automatically caps step size.
        """
        # Check rate limits for budget changes
        if not self.rate_limit_manager.can_change_budget(adset_id):
            raise RuntimeError(f"Ad set budget change limit reached (4/hour) for adset {adset_id}")
        
        b = max(self.cfg.budget_min, min(self.cfg.budget_max, float(daily_budget)))
        if current_budget is not None:
            cap = current_budget * (1.0 + self.cfg.budget_step_cap_pct / 100.0)
            if b > cap:
                b = cap
            jump_pct = ((b - current_budget) / max(1e-9, current_budget)) * 100.0
            if jump_pct > HUMAN_CONFIRM_JUMP_PCT and not human_confirm:
                raise PermissionError(f"Budget jump +{jump_pct:.0f}% exceeds {HUMAN_CONFIRM_JUMP_PCT}%. Set human_confirm=True to proceed.")
        if not self._idempotent_guard(adset_id, "update_budget", {"b": round(b, 2)}):
            return {"skipped": "idempotent"}
        if self.dry_run or not USE_SDK or not self.cfg.enable_budget_updates:
            return {"result": "ok", "mock": True, "action": "update_adset_budget", "adset_id": adset_id, "budget": round(b, 2)}
        self._init_sdk_if_needed()
        self._cooldown()
        
        # Record the budget change attempt
        self.rate_limit_manager.record_budget_change(adset_id)
        
        def _update():
            return AdSet(adset_id).api_update(params={"daily_budget": int(b * 100)})
        return self._retry("update_budget", _update)

    # Back-compat wrapper (USD naming retained)
    def update_adset_budget_usd(
        self,
        adset_id: str,
        daily_budget_usd: float,
        *,
        current_budget_usd: Optional[float] = None,
        human_confirm: bool = False
    ):
        return self.update_adset_budget(
            adset_id,
            daily_budget_usd,
            current_budget=current_budget_usd,
            human_confirm=human_confirm,
        )

    def duplicate_adset(self, adset_id: str, count: int = 1, *, status: str = "PAUSED", prefix: Optional[str] = None, start_time: Optional[str] = None):
        if count <= 0:
            return {"skipped": "count<=0"}
        if not self._idempotent_guard(adset_id, "duplicate", {"count": count, "status": status, "prefix": prefix or ""}):
            return {"skipped": "idempotent"}
        if self.dry_run or not USE_SDK or not self.cfg.enable_duplication:
            ids = [f"{adset_id}_COPY_{i+1}" for i in range(count)]
            return {"result": "ok", "mock": True, "action": "duplicate_adset", "copies": ids}
        self._init_sdk_if_needed()
        self._cooldown()
        params = {"deep_copy": True, "status": status, "count": count}
        if start_time:
            params["start_time"] = start_time
        if prefix:
            params["rename_options"] = {"rename_strategy": "PREFIX_DUPE_NAME", "prefix": prefix}
        def _copy_sdk():
            return AdSet(adset_id).create_ad_set_copy(params=params)
        try:
            return self._retry("duplicate_adset", _copy_sdk)
        except (TypeError, FacebookRequestError):
            payload = dict(params)
            payload["source_adset_id"] = adset_id
            return self._graph_post("adsets", payload)

    # ------------- Ensure (campaign/adset) -------------
    def ensure_campaign(self, name: str, objective: str = "LINK_CLICKS", buying_type: str = "AUCTION") -> Dict[str, Any]:
        self._check_names(campaign=name)
        if self.dry_run or not USE_SDK:
            return {"id": f"CP_{abs(hash(name)) % 10_000_000}", "name": name, "mock": True}
        self._init_sdk_if_needed()

        def _find():
            return AdAccount(self.ad_account_id_act).get_campaigns(fields=["id", "name", "status"], params={"limit": 200})
        camps = self._retry("list_campaigns", _find)
        for c in camps:
            if c.get("name") == name:
                return {"id": c["id"], "name": c.get("name"), "status": c.get("status")}
        self._cooldown()

        def _create_sdk():
            return AdAccount(self.ad_account_id_act).create_campaign(
                fields=[],
                params={"name": _s(name), "objective": _s(objective), "buying_type": buying_type, "status": "PAUSED"},
            )
        try:
            return dict(self._retry("create_campaign", _create_sdk))
        except (TypeError, FacebookRequestError):
            payload = {"name": _s(name), "objective": _s(objective), "buying_type": buying_type, "status": "PAUSED"}
            return self._graph_post("campaigns", payload)

    def ensure_adset(
        self,
        campaign_id: str,
        name: str,
        daily_budget: float,  # in account currency (EUR)
        *,
        optimization_goal: str = "OFFSITE_CONVERSIONS",
        billing_event: str = "IMPRESSIONS",
        bid_strategy: str = "LOWEST_COST_WITHOUT_CAP",
        targeting: Optional[Dict[str, Any]] = None,
        attribution_spec: Optional[List[Dict[str, Any]]] = None,
        placements: Optional[List[str]] = None,   # ✅ NEW
        status: str = "PAUSED",
    ) -> Dict[str, Any]:
        """
        Creates (or returns existing) ad set. Set Instagram/Facebook placements via `placements`,
        e.g. placements=["facebook","instagram"].
        """
        self._check_names(adset=name)
        if self.dry_run or not USE_SDK:
            # simulate targeting with placements applied
            targ = targeting or {"age_min": 18, "genders": [1], "geo_locations": {"countries": ["US"]}}
            if placements:
                targ = dict(targ)
                targ["publisher_platforms"] = placements
                targ["facebook_positions"] = ["feed"] if "facebook" in placements else []
                targ["instagram_positions"] = ["feed", "story", "reels"] if "instagram" in placements else []
            return {
                "id": f"AS_{abs(hash(name)) % 10_000_000}",
                "name": name,
                "mock": True,
                "campaign_id": campaign_id,
                "daily_budget": daily_budget,
                "targeting": targ,
            }

        self._init_sdk_if_needed()

        def _find():
            return Campaign(campaign_id).get_ad_sets(fields=["id", "name", "daily_budget", "status"], params={"limit": 200})
        adsets = self._retry("list_adsets", _find)
        for a in adsets:
            if a.get("name") == name:
                return {"id": a["id"], "name": a.get("name"), "status": a.get("status"), "daily_budget": int(a.get("daily_budget", 0)) / 100.0}

        budget_cents = int(max(self.cfg.budget_min, min(self.cfg.budget_max, daily_budget)) * 100)
        self._cooldown()

        targeting = targeting or {"age_min": 18, "genders": [1], "geo_locations": {"countries": ["US"]}}
        if placements:
            targeting = dict(targeting)
            targeting["publisher_platforms"] = placements
            targeting["facebook_positions"] = ["feed"] if "facebook" in placements else []
            targeting["instagram_positions"] = ["feed", "story", "reels"] if "instagram" in placements else []

        attribution_spec = attribution_spec or [
            {"event_type": "CLICK_THROUGH", "window_days": self.cfg.attribution_click_days},
            {"event_type": "VIEW_THROUGH", "window_days": self.cfg.attribution_view_days},
        ]
        params = {
            "name": _s(name),
            "campaign_id": _s(campaign_id),
            "daily_budget": budget_cents,
            "billing_event": _s(billing_event),
            "optimization_goal": _s(optimization_goal),
            "bid_strategy": _s(bid_strategy),
            "targeting": _sanitize(targeting),
            "status": _s(status),
            "attribution_spec": _sanitize(attribution_spec),
        }

        def _create_sdk():
            return AdAccount(self.ad_account_id_act).create_ad_set(fields=[], params=params)

        try:
            return dict(self._retry("create_adset", _create_sdk))
        except (TypeError, FacebookRequestError):
            return self._graph_post("adsets", params)

    # ------------- Creatives & Ads -------------
    def _get_video_thumbnail_url(self, video_id: str) -> Optional[str]:
        """Best-effort: fetch the preferred thumbnail URI for a video."""
        try:
            res = self._graph_get_object(f"{video_id}/thumbnails", params={"fields": "uri,is_preferred", "limit": 5})
            data = res.get("data") or []
            if not data:
                return None
            for item in data:
                if item.get("is_preferred"):
                    return _s(item.get("uri")).strip()
            return _s(data[0].get("uri")).strip()
        except Exception:
            return None

    def create_video_creative(
        self,
        page_id: Optional[str],
        name: str,
        *,
        video_library_id: Optional[str] = None,
        video_url: Optional[str] = None,  # ignored; we require video_library_id
        primary_text: str,
        headline: str,
        description: str = "",
        call_to_action: str = "SHOP_NOW",
        link_url: Optional[str] = None,
        utm_params: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        instagram_actor_id: Optional[str] = None,   # ✅ NEW
    ) -> Dict[str, Any]:
        """
        Creates a Page video creative. If instagram_actor_id is provided (or IG_ACTOR_ID env is set),
        the creative will be eligible for Instagram placement.
        
        NOTE: Legacy function - ASC+ campaigns use static images via create_image_creative().
        Kept for backward compatibility only.
        """
        self._check_names(ad=_s(name))
        bad = _contains_forbidden([primary_text, headline, description])
        if bad:
            raise ValueError(f"Creative text contains forbidden term: {bad}")

        pid = _s(page_id or os.getenv("FB_PAGE_ID"))
        if not pid:
            raise ValueError("Page ID is required (set FB_PAGE_ID or pass page_id).")

        vid_id = _s(video_library_id).strip()
        if not _is_digits(vid_id):
            raise ValueError(f"Invalid or missing video_library_id '{video_library_id}'. Provide a numeric video ID from the account's Media Library.")

        final_link = _clean_story_link(link_url, utm_params)

        # Ensure we have a thumbnail
        thumb = _s(thumbnail_url).strip()
        if not thumb:
            thumb = self._get_video_thumbnail_url(vid_id)

        ig_id = instagram_actor_id or os.getenv("IG_ACTOR_ID") or None

        if self.dry_run or not USE_SDK or not self.cfg.enable_creative_uploads:
            payload_preview = {
                "page_id": pid,
                "video_id": vid_id,
                "thumbnail_url": thumb or "",
                "instagram_actor_id": ig_id or "",
            }
            return {
                "id": f"CR_{abs(hash(_s(name))) % 10_000_000}",
                "name": _s(name),
                "mock": True,
                **payload_preview,
            }

        self._init_sdk_if_needed()
        self._cooldown()

        video_data: Dict[str, Any] = {
            "message": _s(primary_text),
            "video_id": vid_id,
        }
        if headline:
            video_data["link_description"] = _s(headline)[:100]
        if final_link:
            video_data["call_to_action"] = {"type": _s(call_to_action or "SHOP_NOW"), "value": {"link": _s(final_link)}}
        if thumb:
            video_data["image_url"] = _s(thumb)

        story_spec: Dict[str, Any] = {"page_id": pid, "video_data": video_data}
        if ig_id:
            story_spec["instagram_actor_id"] = ig_id  # ✅ enable IG placement

        params = {
            "name": _s(name),
            "object_story_spec": story_spec,
        }

        # HTTP-first; SDK fallback
        try:
            creative = self._graph_post("adcreatives", params)
        except Exception:
            def _create_sdk():
                return AdAccount(self.ad_account_id_act).create_ad_creative(fields=[], params=_sanitize(params))
            creative = dict(self._retry("create_creative", _create_sdk))

        # Verify video attached
        cid = creative.get("id")
        if not cid:
            raise RuntimeError("Video creative verification failed: creative has no ID.")
        try:
            fetched = self._graph_get_object(f"{cid}", params={"fields": "object_story_spec"})
            oss = fetched.get("object_story_spec") or {}
        except Exception:
            fetched = AdCreative(cid).api_get(fields=["object_story_spec"])
            oss = fetched.get("object_story_spec") or {}
        vdat = (oss.get("video_data") or {}) if isinstance(oss, dict) else {}
        attached = _s(vdat.get("video_id")).strip()
        if attached != vid_id:
            raise RuntimeError(f"Video creative verification failed: video mismatch (expected={vid_id!r}, got={attached!r})")

        return creative

    def create_image_creative(
        self,
        page_id: Optional[str],
        name: str,
        *,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,  # Local file path - will use Supabase Storage
        supabase_storage_url: Optional[str] = None,  # Supabase Storage URL (preferred)
        primary_text: str,
        headline: str,
        description: str = "",
        call_to_action: str = "SHOP_NOW",
        link_url: Optional[str] = None,
        utm_params: Optional[str] = None,
        instagram_actor_id: Optional[str] = None,
        creative_id: Optional[str] = None,  # Creative ID for tracking
    ) -> Dict[str, Any]:
        """
        Creates a static image creative for Meta Ads.
        Uses Supabase Storage URL if provided, otherwise falls back to local upload.
        """
        self._check_names(ad=_s(name))
        bad = _contains_forbidden([primary_text, headline, description])
        if bad:
            raise ValueError(f"Creative text contains forbidden term: {bad}")

        pid = _s(page_id or os.getenv("FB_PAGE_ID"))
        if not pid:
            raise ValueError("Page ID is required (set FB_PAGE_ID or pass page_id).")

        # Priority: supabase_storage_url > image_url > image_path
        final_image_url = _s(supabase_storage_url or image_url).strip()
        
        if not final_image_url and image_path:
            # Try to get Supabase Storage URL for this creative
            if creative_id:
                try:
                    from infrastructure.creative_storage import create_creative_storage_manager
                    from infrastructure.supabase_storage import get_validated_supabase_client
                    
                    supabase_client = get_validated_supabase_client()
                    if supabase_client:
                        storage_manager = create_creative_storage_manager(supabase_client)
                        if storage_manager:
                            storage_url = storage_manager.get_creative_url(creative_id)
                            if storage_url:
                                final_image_url = storage_url
                                # Update usage
                                storage_manager.update_usage(creative_id)
                            else:
                                # Upload if not in storage
                                final_image_url = storage_manager.upload_creative(
                                    creative_id=creative_id,
                                    image_path=image_path,
                                )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to get/create Supabase Storage URL: {e}")
            
            # Fallback: upload to Meta's ad image library (legacy)
            if not final_image_url:
                try:
                    uploaded_image = self._upload_ad_image(image_path, name)
                    if uploaded_image and uploaded_image.get("hash"):
                        final_image_url = uploaded_image.get("hash")
                    elif uploaded_image and uploaded_image.get("url"):
                        final_image_url = uploaded_image.get("url")
                except Exception as e:
                    notify(f"⚠️ Failed to upload image, using path as URL: {e}")
                    final_image_url = f"file://{image_path}"

        if not final_image_url:
            raise ValueError("Either supabase_storage_url, image_url, or image_path must be provided.")

        final_link = _clean_story_link(link_url, utm_params)
        ig_id = instagram_actor_id or os.getenv("IG_ACTOR_ID") or None
        # Only use instagram_actor_id if it's a valid non-empty string
        # CRITICAL: If the ID is invalid (causes API errors), don't use it
        if ig_id:
            ig_id = str(ig_id).strip()
            if not ig_id:
                ig_id = None
            # Validate it's a numeric string (Instagram IDs are numeric)
            elif not ig_id.isdigit():
                logger.warning(f"Invalid Instagram actor ID format (not numeric): {ig_id} - skipping")
                ig_id = None
            # Additional validation: Check if it's a known invalid ID that causes errors
            elif ig_id == "17841477094913251":  # This specific ID causes errors
                logger.warning(f"Known invalid Instagram actor ID: {ig_id} - skipping")
                ig_id = None

        if self.dry_run or not USE_SDK or not self.cfg.enable_creative_uploads:
            payload_preview = {
                "page_id": pid,
                "image_url": final_image_url,
            }
            if ig_id:
                payload_preview["instagram_actor_id"] = ig_id
            return {
                "id": f"CR_{abs(hash(_s(name))) % 10_000_000}",
                "name": _s(name),
                "mock": True,
                **payload_preview,
            }

        self._init_sdk_if_needed()
        self._cooldown()

        image_data: Dict[str, Any] = {
            "message": _s(primary_text),  # Primary text (main ad copy)
            "link": _s(final_link) if final_link else os.getenv("SHOPIFY_STORE_URL", "https://brava-skin.com"),  # REQUIRED: link field
        }
        
        # CRITICAL: Meta API doesn't accept image_url in link_data for single image creatives
        # We must upload the image to Meta's Ad Image library first and use image_hash
        if final_image_url.startswith("http"):
            # Upload image to Meta's Ad Image library
            try:
                import requests
                import tempfile
                import os
                
                logger.info(f"Uploading image to Meta Ad Image library: {final_image_url}")
                
                # Download image from URL
                img_response = requests.get(final_image_url, timeout=30)
                img_response.raise_for_status()
                
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(img_response.content)
                
                try:
                    # Upload to Meta
                    # Get API version from account or default to v23.0
                    api_version = getattr(self.account, 'api_version', 'v23.0') if hasattr(self, 'account') else 'v23.0'
                    # Remove 'v' prefix if present
                    api_version = api_version.replace('v', '') if api_version.startswith('v') else api_version
                    upload_url = f"https://graph.facebook.com/v{api_version}/act_{self.ad_account_id_act.replace('act_', '')}/adimages"
                    files = {"source": open(tmp_path, "rb")}
                    # Get access token from account
                    access_token = getattr(self.account, 'access_token', None) if hasattr(self, 'account') else None
                    if not access_token:
                        # Fallback: try to get from environment or config
                        access_token = os.getenv("FB_ACCESS_TOKEN") or os.getenv("FACEBOOK_ACCESS_TOKEN")
                    if not access_token:
                        raise RuntimeError("No access token available for image upload")
                    upload_data = {"access_token": access_token}
                    
                    upload_response = requests.post(upload_url, files=files, data=upload_data, timeout=30)
                    files["source"].close()
                    
                    if upload_response.status_code != 200:
                        error_data = upload_response.json() if upload_response.text else {}
                        error_msg = error_data.get("error", {}).get("message", "Unknown error")
                        raise RuntimeError(f"Meta image upload failed: {error_msg}")
                    
                    result = upload_response.json()
                    images = result.get("images", {})
                    if images:
                        # Get first hash - the key is the hash
                        image_hash = list(images.keys())[0]
                        # Also verify the hash value inside
                        hash_data = images[image_hash]
                        actual_hash = hash_data.get("hash", image_hash)
                        logger.info(f"✅ Uploaded image to Meta, hash: {actual_hash}")
                        image_data["image_hash"] = actual_hash
                    else:
                        raise ValueError(f"No image hash in upload response: {result}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception as e:
                logger.error(f"Failed to upload image to Meta: {e}, falling back to image_url (may fail)")
                # Fallback to image_url (will likely fail, but try anyway)
                image_data["image_url"] = _s(final_image_url)
        else:
            # Already a hash
            image_data["image_hash"] = _s(final_image_url)
        
        # Headline goes in "name" field for single image creatives
        if headline:
            image_data["name"] = _s(headline)[:100]
        
        # Description (optional)
        if description:
            image_data["description"] = _s(description)[:150]
        
        # "Shop now" CTA
        if final_link:
            image_data["call_to_action"] = {"type": _s(call_to_action or "SHOP_NOW"), "value": {"link": _s(final_link)}}

        story_spec: Dict[str, Any] = {"page_id": pid, "link_data": image_data}
        if ig_id:
            story_spec["instagram_actor_id"] = ig_id

        params = {
            "name": _s(name),
            "object_story_spec": story_spec,
        }

        # HTTP-first; SDK fallback
        try:
            creative = self._graph_post("adcreatives", params)
        except Exception:
            def _create_sdk():
                return AdAccount(self.ad_account_id_act).create_ad_creative(fields=[], params=_sanitize(params))
            creative = dict(self._retry("create_creative", _create_sdk))

        return creative

    def create_carousel_creative(
        self,
        page_id: Optional[str],
        name: str,
        *,
        main_image_url: Optional[str] = None,
        main_image_path: Optional[str] = None,
        supabase_storage_url: Optional[str] = None,
        primary_text: str,
        headline: str,
        description: str = "",
        call_to_action: str = "SHOP_NOW",
        link_url: Optional[str] = None,
        utm_params: Optional[str] = None,
        instagram_actor_id: Optional[str] = None,
        product_catalog_id: Optional[str] = None,  # Catalog ID for product cards
        product_set_id: Optional[str] = None,  # Product set ID (optional, uses catalog if not provided)
    ) -> Dict[str, Any]:
        """
        Creates a carousel creative with main image and catalog products.
        Main image is the generated creative, additional cards are from catalog.
        """
        self._check_names(ad=_s(name))
        bad = _contains_forbidden([primary_text, headline, description])
        if bad:
            raise ValueError(f"Creative text contains forbidden term: {bad}")

        pid = _s(page_id or os.getenv("FB_PAGE_ID"))
        if not pid:
            raise ValueError("Page ID is required (set FB_PAGE_ID or pass page_id).")

        # Get main image URL (priority: supabase_storage_url > main_image_url > main_image_path)
        final_image_url = _s(supabase_storage_url or main_image_url).strip()
        
        if not final_image_url and main_image_path:
            # Upload to Meta's ad image library
            try:
                uploaded_image = self._upload_ad_image(main_image_path, name)
                if uploaded_image and uploaded_image.get("hash"):
                    final_image_url = uploaded_image.get("hash")
                elif uploaded_image and uploaded_image.get("url"):
                    final_image_url = uploaded_image.get("url")
            except Exception as e:
                notify(f"⚠️ Failed to upload image: {e}")
                raise

        if not final_image_url:
            raise ValueError("Either supabase_storage_url, main_image_url, or main_image_path must be provided.")

        final_link = _clean_story_link(link_url, utm_params)
        ig_id = instagram_actor_id or os.getenv("IG_ACTOR_ID") or None

        # Get catalog ID from ad set or env
        catalog_id = product_catalog_id or os.getenv("PRODUCT_CATALOG_ID")
        if not catalog_id:
            logger.warning("No product catalog ID provided - carousel will only show main image")

        if self.dry_run or not USE_SDK or not self.cfg.enable_creative_uploads:
            payload_preview = {
                "page_id": pid,
                "carousel": True,
                "main_image_url": final_image_url,
                "catalog_id": catalog_id or "",
                "instagram_actor_id": ig_id or "",
            }
            return {
                "id": f"CR_{abs(hash(_s(name))) % 10_000_000}",
                "name": _s(name),
                "mock": True,
                **payload_preview,
            }

        self._init_sdk_if_needed()
        self._cooldown()

        # Upload main image to get hash
        main_image_hash = None
        if final_image_url.startswith("http"):
            # Need to upload to get hash for carousel
            try:
                import requests
                import tempfile
                response = requests.get(final_image_url, timeout=30)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                uploaded = self._upload_ad_image(tmp_path, f"{name}_main")
                if uploaded and uploaded.get("hash"):
                    main_image_hash = uploaded.get("hash")
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to upload main image for carousel: {e}")
                # Fallback to URL
                main_image_hash = None
        else:
            main_image_hash = final_image_url  # Already a hash

        # Build carousel child attachments
        # First attachment: main image (our generated creative)
        main_attachment: Dict[str, Any] = {
            "link": final_link or os.getenv("SHOPIFY_STORE_URL", "https://brava-skin.com"),
        }
        
        if headline:
            main_attachment["name"] = _s(headline)[:100]
        if description:
            main_attachment["description"] = _s(description)[:150]
        
        if main_image_hash:
            main_attachment["image_hash"] = main_image_hash
        else:
            main_attachment["image_url"] = final_image_url

        child_attachments = [main_attachment]

        # For carousel with catalog, Meta will automatically add product cards from the catalog
        # configured in the ad set. We just need to create a carousel format.
        story_spec: Dict[str, Any] = {
            "page_id": pid,
            "link_data": {
                "message": _s(primary_text),
                "name": _s(headline)[:100] if headline else "",
                "description": _s(description)[:150] if description else "",
                "link": final_link or os.getenv("SHOPIFY_STORE_URL", "https://brava-skin.com"),
                "call_to_action": {"type": _s(call_to_action or "SHOP_NOW"), "value": {"link": final_link or os.getenv("SHOPIFY_STORE_URL", "https://brava-skin.com")}},
                "child_attachments": child_attachments,
            }
        }

        # Note: Catalog products are automatically added by Meta from the ad set's catalog configuration
        # We don't need to specify them here - Meta will pull products from the catalog chosen in the ad set

        if ig_id:
            story_spec["instagram_actor_id"] = ig_id

        params = {
            "name": _s(name),
            "object_story_spec": story_spec,
        }

        # HTTP-first; SDK fallback
        try:
            creative = self._graph_post("adcreatives", params)
        except Exception:
            def _create_sdk():
                return AdAccount(self.ad_account_id_act).create_ad_creative(fields=[], params=_sanitize(params))
            creative = dict(self._retry("create_creative", _create_sdk))

        return creative

    def _upload_ad_image(self, image_path: str, name: str) -> Dict[str, Any]:
        """
        Upload an image to Meta's Ad Image library.
        Returns the uploaded image hash or URL.
        """
        if self.dry_run or not USE_SDK:
            return {"hash": f"MOCK_{abs(hash(image_path)) % 10_000_000}", "mock": True}

        self._init_sdk_if_needed()
        self._cooldown()

        try:
            # Read image file
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Upload using AdAccount.create_ad_image
            def _upload():
                return AdAccount(self.ad_account_id_act).create_ad_image(
                    files={"file": image_data},
                    params={"name": _s(name)}
                )
            
            result = self._retry("upload_image", _upload)
            return dict(result) if hasattr(result, "__dict__") else result
            
        except Exception as e:
            notify(f"❌ Failed to upload image: {e}")
            raise

    def create_ad(
        self, 
        adset_id: str, 
        name: str, 
        creative_id: str, 
        status: str = "PAUSED", 
        *, 
        original_ad_id: Optional[str] = None,
        instagram_actor_id: Optional[str] = None,  # Instagram account at ad level (alternative approach)
        tracking_specs: Optional[List[Dict[str, Any]]] = None,  # For shop destination/conversion tracking
    ) -> Dict[str, Any]:
        self._check_names(ad=_s(name))

        payload = {
            "name": _s(name),
            "adset_id": _s(adset_id),
            "creative": {"creative_id": _s(creative_id)},
            "status": _s(status),
        }
        
        # Add Instagram actor ID at ad level (alternative approach if creative level doesn't work)
        ig_id = instagram_actor_id or os.getenv("IG_ACTOR_ID") or None
        if ig_id:
            ig_id = str(ig_id).strip()
            if ig_id and ig_id.isdigit() and ig_id != "17841477094913251":  # Valid Instagram ID
                payload["instagram_actor_id"] = ig_id
                logger.info(f"Adding Instagram actor ID at ad level: {ig_id}")
            else:
                logger.warning(f"Invalid Instagram actor ID format: {ig_id} - skipping")
        
        # Add tracking specs for shop destination/conversion tracking
        # For Advantage+ Shopping Campaigns, shop destination is typically set at ad set level
        # But we can also add it at ad level for explicit control
        if tracking_specs:
            payload["tracking_specs"] = tracking_specs
            logger.info(f"Adding tracking specs for shop destination: {tracking_specs}")
        else:
            # For shop destination, we might need conversion_specs instead of tracking_specs
            # But tracking_specs with pixel is also valid for conversion tracking
            pixel_id = os.getenv("FB_PIXEL_ID")
            if pixel_id:
                # Add tracking specs for purchase conversions (shop destination)
                # Meta API requires "action.type" (with dot), not "action_type" (with underscore)
                payload["tracking_specs"] = [
                    {
                        "action.type": "purchase",
                        "fb_pixel": [pixel_id]
                    }
                ]
                logger.info(f"Added tracking specs with pixel ID for shop destination: {pixel_id}")
            
            # Note: Shop destination is primarily controlled by:
            # 1. Ad set level: catalog_id and conversion_location
            # 2. Campaign objective: SALES with Advantage+ Shopping Campaign
            # The tracking_specs here are for conversion tracking, not destination selection

        if self.dry_run or not self.cfg.enable_creative_uploads:
            # Use original_ad_id if provided for ID continuity, otherwise generate new ID
            ad_id = original_ad_id if original_ad_id else f"AD_{abs(hash(_s(name))) % 10_000_000}"
            return {"id": ad_id, "name": _s(name), "mock": True, "status": status, **payload}

        self._cooldown()
        try:
            return self._graph_post("ads", payload)
        except Exception:
            if not USE_SDK:
                raise
            self._init_sdk_if_needed()
            def _create_sdk():
                return AdAccount(self.ad_account_id_act).create_ad(fields=[], params=_sanitize(payload))
            return dict(self._retry("create_ad", _create_sdk))

    def promote_ad_with_continuity(self, original_ad_id: str, new_adset_id: str, new_name: str, creative_id: str, status: str = "ACTIVE") -> Dict[str, Any]:
        """
        Promote an ad to a new adset while maintaining the same ID for continuity.
        This is used when moving ads between stages (TEST->VALID->SCALE).
        """
        self._check_names(ad=_s(new_name))

        payload = {
            "name": _s(new_name),
            "adset_id": _s(new_adset_id),
            "creative": {"creative_id": _s(creative_id)},
            "status": _s(status),
        }

        if self.dry_run or not self.cfg.enable_creative_uploads:
            return {"id": original_ad_id, "name": _s(new_name), "mock": True, "status": status, "promoted_from": original_ad_id}

        self._cooldown()
        try:
            # For real API calls, we need to create a new ad but track the relationship
            result = self._graph_post("ads", payload)
            # Add the original ID to the result for tracking purposes
            result["promoted_from"] = original_ad_id
            return result
        except Exception:
            if not USE_SDK:
                raise
            self._init_sdk_if_needed()
            def _create_sdk():
                return AdAccount(self.ad_account_id_act).create_ad(fields=[], params=_sanitize(payload))
            result = dict(self._retry("create_ad", _create_sdk))
            result["promoted_from"] = original_ad_id
            return result

    # ------------- Convenience (budgets in EUR now) -------------
    def create_validation_adset(self, campaign_id: str, creative_label: str, daily_budget: float = 40.0, *, placements: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.ensure_adset(_s(campaign_id), f"[VALID] {_s(creative_label)}", daily_budget, placements=placements)

    def create_scaling_adset(self, campaign_id: str, creative_label: str, daily_budget: float = 100.0, *, placements: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.ensure_adset(_s(campaign_id), f"[SCALE] {_s(creative_label)}", daily_budget, placements=placements)

    # ------------- Data quality & reconciliation -------------
    @staticmethod
    def reconcile_roas(row: Dict[str, Any]) -> Tuple[float, float]:
        spend = float(row.get("spend") or 0.0)
        rev = 0.0
        for v in (row.get("action_values") or []):
            if v.get("action_type") == "purchase":
                try:
                    rev += float(v.get("value", 0))
                except Exception:
                    pass
        roas_computed = (rev / spend) if spend > 0 else 0.0
        roas_field = 0.0
        roas_list = row.get("purchase_roas") or []
        if isinstance(roas_list, list) and roas_list:
            try:
                roas_field = float(roas_list[0].get("value", 0))
            except Exception:
                roas_field = 0.0
        return roas_computed, roas_field

    # ------------- Dry-run planner -------------
    def plan_budget_change(self, adset_id: str, current_budget: float, target_budget: float) -> Dict[str, Any]:
        """
        Plans a budget change in account currency (EUR).
        """
        target = max(self.cfg.budget_min, min(self.cfg.budget_max, float(target_budget)))
        cap = current_budget * (1.0 + self.cfg.budget_step_cap_pct / 100.0)
        step = min(target, cap)
        jump_pct = ((step - current_budget) / max(1e-9, current_budget)) * 100.0
        requires_human = jump_pct > HUMAN_CONFIRM_JUMP_PCT
        return {"adset_id": _s(adset_id), "current": current_budget, "target": target, "first_step": step, "jump_pct": jump_pct, "requires_human": requires_human}

    # ------------- Name search -------------
    def find_ad_by_name(self, name: str) -> Optional[str]:
        if self.dry_run or not USE_SDK:
            return None
        self._init_sdk_if_needed()
        def _fetch():
            return AdAccount(self.ad_account_id_act).get_ads(fields=["id", "name"], params={"limit": 500})
        ads = self._retry("find_ad_by_name", _fetch)
        for a in ads:
            if a.get("name") == _s(name):
                return a["id"]
        return None

    # ------------- Preflight checks (best-effort) -------------
    def preflight(self) -> Dict[str, Any]:
        if self.dry_run or not USE_SDK:
            return {"ok": True, "dry_run": True}
        ok, issues = True, []
        try:
            def _me():
                return AdAccount(self.ad_account_id_act).api_get(fields=["account_id", "currency", "timezone_name", "account_status"])
            info = self._retry("preflight_get", _me)
            if str(info.get("account_status")) not in ("1", "2"):
                ok, issues = False, issues + [f"Account status={info.get('account_status')}"]
            # soft sanity: expected EUR + Europe/Amsterdam (allow partial mismatch without failing)
            tz = (info.get("timezone_name") or "").lower()
            ccy = (info.get("currency") or "").upper()
            if "amsterdam" not in tz:
                issues.append(f"Note: account timezone is {info.get('timezone_name')}")
            if ccy != self.cfg.currency.upper():
                issues.append(f"Note: account currency is {info.get('currency')}")
        except Exception as e:
            ok, issues = False, issues + [f"preflight error: {e}"]
        return {"ok": ok, "issues": issues}

    def _get_billing_details(self) -> Optional[Dict[str, Any]]:
        """
        Get billing details including auto-charge threshold from Meta's billing API.
        Returns billing information if available, None if not accessible.
        """
        if self.dry_run or not USE_SDK:
            return None
        
        try:
            # Try to get billing information from the account
            # Note: This may require additional permissions or may not be available for all account types
            def _get_billing_info():
                return AdAccount(self.ad_account_id_act).api_get(fields=[
                    "funding_source", "billing_center", "payment_methods"
                ])
            
            billing_info = self._retry("billing_details", _get_billing_info)
            
            # Try to extract auto-charge threshold from funding source
            funding_source = billing_info.get("funding_source", {})
            if isinstance(funding_source, dict):
                # Look for auto-charge threshold in various possible fields
                auto_charge_threshold = (
                    funding_source.get("auto_charge_threshold") or
                    funding_source.get("threshold") or
                    funding_source.get("min_balance") or
                    funding_source.get("recharge_threshold")
                )
                
                if auto_charge_threshold is not None:
                    return {
                        "auto_charge_threshold": auto_charge_threshold,
                        "funding_source_type": funding_source.get("type", "unknown"),
                        "payment_method_status": funding_source.get("status", "unknown")
                    }
            
            # If no auto-charge threshold found, return basic billing info
            return {
                "funding_source_type": funding_source.get("type", "unknown"),
                "payment_method_status": funding_source.get("status", "unknown"),
                "auto_charge_threshold": None
            }
            
        except Exception:
            # Billing details are optional, return None if not accessible
            return None

    def check_account_health(self) -> Dict[str, Any]:
        """
        Comprehensive ad account health check including payment status, billing issues, and account restrictions.
        Returns detailed health information for monitoring and alerting.
        """
        if self.dry_run or not USE_SDK:
            return {"ok": True, "dry_run": True, "health_details": {}}
        
        health_details = {}
        critical_issues = []
        warnings = []
        
        try:
            # Get comprehensive account information
            def _get_account_info():
                return AdAccount(self.ad_account_id_act).api_get(fields=[
                    "account_id", "currency", "timezone_name", "account_status", 
                    "amount_spent", "balance", "spend_cap", "funding_source",
                    "business_name", "business_country_code", "business_zip"
                ])
            
            account_info = self._retry("account_health", _get_account_info)
            health_details["account_info"] = account_info
            
            # Check account status
            account_status = str(account_info.get("account_status", ""))
            if account_status not in ("1", "2"):  # 1=Active, 2=Active (Limited)
                critical_issues.append(f"Account status is {account_status} (not active)")
                health_details["account_status"] = "inactive"
            else:
                health_details["account_status"] = "active"
            
            # Check for payment/billing issues and get auto-charge threshold
            funding_source = account_info.get("funding_source")
            if funding_source:
                health_details["funding_source"] = funding_source
                # Check if funding source indicates payment issues
                if isinstance(funding_source, dict):
                    source_type = funding_source.get("type", "").lower()
                    if "credit_card" in source_type or "payment_method" in source_type:
                        # Check if there are any payment method issues
                        if funding_source.get("status", "").lower() in ["failed", "declined", "expired"]:
                            critical_issues.append(f"Payment method issue: {funding_source.get('status')}")
                            health_details["payment_status"] = "failed"
                        else:
                            health_details["payment_status"] = "ok"
            
            # Try to get billing details including auto-charge threshold
            try:
                billing_details = self._get_billing_details()
                if billing_details:
                    health_details["billing_details"] = billing_details
                    # Extract auto-charge threshold if available
                    auto_charge_threshold = billing_details.get("auto_charge_threshold")
                    if auto_charge_threshold is not None:
                        health_details["auto_charge_threshold"] = float(auto_charge_threshold) / 100.0  # Convert from cents
            except Exception as e:
                # Billing details are optional, don't fail the health check
                health_details["billing_error"] = str(e)
            
            # Check account balance and spending limits
            balance = account_info.get("balance")
            spend_cap = account_info.get("spend_cap")
            amount_spent = account_info.get("amount_spent")
            
            if balance is not None:
                health_details["balance"] = float(balance) / 100.0  # Convert from cents
                if float(balance) <= 0:
                    warnings.append("Account balance is zero or negative")
            
            if spend_cap is not None:
                spend_cap_float = float(spend_cap) / 100.0
                health_details["spend_cap"] = spend_cap_float
                if amount_spent is not None:
                    spent_amount = float(amount_spent) / 100.0
                    health_details["amount_spent"] = spent_amount
                    if spend_cap_float > 0 and spent_amount >= spend_cap_float * 0.9:  # 90% of spend cap
                        warnings.append(f"Account has spent {spent_amount:.2f} of {spend_cap_float:.2f} spend cap")
            
            # Check for business verification issues
            business_name = account_info.get("business_name")
            business_country = account_info.get("business_country_code")
            business_zip = account_info.get("business_zip")
            
            if not business_name:
                warnings.append("Business name not set - may affect ad delivery")
            if not business_country:
                warnings.append("Business country not set - may affect ad delivery")
            if not business_zip:
                warnings.append("Business ZIP not set - may affect ad delivery")
            
            health_details["business_verification"] = {
                "name": bool(business_name),
                "country": bool(business_country),
                "zip": bool(business_zip)
            }
            
        except Exception as e:
            critical_issues.append(f"Account health check failed: {str(e)}")
            health_details["error"] = str(e)
        
        # Determine overall health status
        if critical_issues:
            health_details["overall_status"] = "critical"
        elif warnings:
            health_details["overall_status"] = "warning"
        else:
            health_details["overall_status"] = "healthy"
        
        health_details["critical_issues"] = critical_issues
        health_details["warnings"] = warnings
        
        return {
            "ok": len(critical_issues) == 0,
            "health_details": health_details,
            "critical_issues": critical_issues,
            "warnings": warnings
        }
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get comprehensive rate limiting status information.
        """
        return self.rate_limit_manager.get_status()


__all__ = ["AccountAuth", "ClientConfig", "MetaClient"]
