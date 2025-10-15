# meta_client.py
from __future__ import annotations

import hashlib
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import requests

from utils import today_ymd, yesterday_ymd

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
META_RETRY_MAX          = int(os.getenv("META_RETRY_MAX", "4") or 4)
META_BACKOFF_BASE       = float(os.getenv("META_BACKOFF_BASE", "0.4") or 0.4)
META_TIMEOUT            = float(os.getenv("META_TIMEOUT", "30") or 30)
META_WRITE_COOLDOWN_SEC = int(os.getenv("META_WRITE_COOLDOWN_SEC", "5") or 5)

BUDGET_MIN          = float(os.getenv("BUDGET_MIN", "5") or 5.0)
BUDGET_MAX          = float(os.getenv("BUDGET_MAX", "50000") or 50000.0)
BUDGET_MAX_STEP_PCT = float(os.getenv("BUDGET_MAX_STEP_PCT", "200") or 200.0)

# Circuit breaker
CB_FAILS     = int(os.getenv("META_CB_FAILS", "5") or 5)
CB_RESET_SEC = int(os.getenv("META_CB_RESET_SEC", "120") or 120)

# Naming & compliance
CAMPAIGN_NAME_RE = re.compile(r"^\[(TEST|VALID|SCALE|SCALE-CBO)\]\s+Brava\s+—\s+(ABO|CBO)\s+—\s+US Men$")
ADSET_NAME_RE    = re.compile(r"^\[(TEST|VALID|SCALE)\]\s+.+$")
AD_NAME_RE       = re.compile(r"^\[(TEST|VALID|SCALE)\]\s+.+$")
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
                    try:
                        if "Retry-After" in headers:
                            retry_after = float(headers.get("Retry-After"))
                    except Exception:
                        retry_after = None
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
            except Exception:
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

    def _graph_post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._graph_url(endpoint)
        data = _sanitize(payload)
        data["access_token"] = self.account.access_token
        r = requests.post(url, json=data, timeout=META_TIMEOUT)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"error": {"message": r.text}}
            msg = f"Graph POST {endpoint} {r.status_code}: {err}"
            try:
                code = err.get("error", {}).get("code")
                sub = err.get("error", {}).get("error_subcode")
                if code == 100 and sub == 33:
                    msg += " — Hint: check ad account id, token scopes (ads_management), and account access."
            except Exception:
                pass
            raise RuntimeError(msg)
        try:
            return r.json()
        except Exception:
            return {"ok": True, "text": r.text}

    def _graph_get_object(self, object_path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET for absolute objects like '/{id}' or '/{id}/thumbnails' (not under ad account path)."""
        ver = self.account.api_version or "v23.0"
        url = f"https://graph.facebook.com/{ver}/{object_path.lstrip('/')}"
        qp = dict(params or {})
        qp["access_token"] = self.account.access_token
        r = requests.get(url, params=qp, timeout=META_TIMEOUT)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"error": {"message": r.text}}
            raise RuntimeError(f"Graph GET {object_path} {r.status_code}: {err}")
        return r.json()

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

    def create_ad(self, adset_id: str, name: str, creative_id: str, status: str = "PAUSED", *, original_ad_id: Optional[str] = None) -> Dict[str, Any]:
        self._check_names(ad=_s(name))

        payload = {
            "name": _s(name),
            "adset_id": _s(adset_id),
            "creative": {"creative_id": _s(creative_id)},
            "status": _s(status),
        }

        if self.dry_run or not self.cfg.enable_creative_uploads:
            # Use original_ad_id if provided for ID continuity, otherwise generate new ID
            ad_id = original_ad_id if original_ad_id else f"AD_{abs(hash(_s(name))) % 10_000_000}"
            return {"id": ad_id, "name": _s(name), "mock": True, "status": status}

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


__all__ = ["AccountAuth", "ClientConfig", "MetaClient"]
