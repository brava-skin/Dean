# meta_client.py
from __future__ import annotations

import json
import os
import time
import hashlib
import hmac
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Try the official SDK, fall back to raw HTTP
USE_SDK = False
try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.adobjects.adset import AdSet
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.ad import Ad
    USE_SDK = True
except Exception:
    pass

# ------------------------- CONFIG -------------------------
META_TIMEOUT = float(os.getenv("META_TIMEOUT", "30"))
META_RETRY_MAX = int(os.getenv("META_RETRY_MAX", "4"))
META_BACKOFF_BASE = float(os.getenv("META_BACKOFF_BASE", "0.4"))
META_WRITE_COOLDOWN_SEC = int(os.getenv("META_WRITE_COOLDOWN_SEC", "5"))

BUDGET_MIN = float(os.getenv("BUDGET_MIN", "5"))
BUDGET_MAX = float(os.getenv("BUDGET_MAX", "50000"))
BUDGET_STEP_PCT = float(os.getenv("BUDGET_MAX_STEP_PCT", "200"))
ACCOUNT_TZ = os.getenv("ACCOUNT_TZ", "Europe/Amsterdam")
ACCOUNT_CCY = os.getenv("ACCOUNT_CURRENCY", "EUR")

FORBIDDEN_TERMS = tuple(
    t.strip().lower() for t in os.getenv("FORBIDDEN_TERMS", "cures,miracle,guaranteed").split(",")
)

# ------------------------- HELPERS -------------------------
def _sanitize(obj: Any) -> Any:
    """Recursively sanitize payloads to JSON-serializable structures."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return str(obj)

def _contains_forbidden(texts: List[Optional[str]]) -> Optional[str]:
    for t in texts:
        low = (t or "").lower()
        for bad in FORBIDDEN_TERMS:
            if bad and bad in low:
                return bad
    return None

def _normalize_account_id(account_id: str) -> Tuple[str, str]:
    aid = (account_id or "").strip()
    num = aid[4:] if aid.startswith("act_") else aid
    return num, f"act_{num}"

def _encode_param_value(key: str, value: Any) -> Any:
    """
    Encode parameters the way Graph expects.
    - fields: comma string
    - breakdowns/action_breakdowns: comma string or JSON list is accepted; we use comma string
    - action_attribution_windows: JSON list string is safest
    - filtering/time_range: JSON string
    - generic lists: comma string
    """
    if value is None:
        return None

    # Keys that should be JSON-encoded objects/arrays
    if key in ("filtering", "time_range"):
        return json.dumps(value)

    # Keys that should be JSON-encoded lists (per docs)
    if key in ("action_attribution_windows",):
        return json.dumps(value)

    # Keys that prefer comma-joined strings
    if key in ("fields", "breakdowns", "action_breakdowns"):
        if isinstance(value, (list, tuple)):
            return ",".join(map(str, value))
        return str(value)

    # Generic lists -> comma-separated
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))

    return value

def _appsecret_proof(app_secret: str, access_token: str) -> str:
    return hmac.new(
        app_secret.encode("utf-8"),
        msg=access_token.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

# ------------------------- DATACLASSES -------------------------
@dataclass
class AccountAuth:
    account_id: str
    access_token: str
    app_id: str
    app_secret: str
    api_version: str = "v24.0"  # default to latest you target

@dataclass
class ClientConfig:
    timezone: str = ACCOUNT_TZ
    currency: str = ACCOUNT_CCY
    budget_min: float = BUDGET_MIN
    budget_max: float = BUDGET_MAX
    budget_step_cap_pct: float = BUDGET_STEP_PCT
    write_cooldown_sec: int = META_WRITE_COOLDOWN_SEC
    enable_creative_uploads: bool = True
    enable_budget_updates: bool = True
    enable_duplication: bool = True
    require_name_compliance: bool = True

# ------------------------- MAIN CLIENT -------------------------
class MetaClient:
    """
    Minimal-but-robust Meta Marketing API client with:
    - Raw HTTP Graph calls (SDK optional)
    - Insights helpers (levels, fields, breakdowns, attribution, filtering, sort, pagination)
    - Async insights job support
    - Simple write helpers for adsets/creatives/ads
    """

    def __init__(
        self,
        accounts: Union[List[AccountAuth], AccountAuth],
        cfg: Optional[ClientConfig] = None,
        *,
        dry_run: bool = True,
        store=None,
        tenant_id: str = "default",
    ):
        self.cfg = cfg or ClientConfig()
        self.dry_run = dry_run
        self.store = store
        self.tenant_id = tenant_id

        self.accounts = accounts if isinstance(accounts, list) else [accounts]
        self._acct_num: Dict[int, str] = {}
        self._acct_act: Dict[int, str] = {}
        for i, acc in enumerate(self.accounts):
            num, act = _normalize_account_id(acc.account_id)
            self._acct_num[i], self._acct_act[i] = num, act

        self._active = 0
        self._last_write_ts = 0
        self._init_sdk()

    # ---- Core props ----
    @property
    def account(self) -> AccountAuth:
        return self.accounts[self._active]

    @property
    def act_id(self) -> str:
        return self._acct_act[self._active]

    def _init_sdk(self):
        if USE_SDK and not self.dry_run:
            a = self.account
            FacebookAdsApi.init(a.app_id, a.app_secret, a.access_token, api_version=a.api_version)

    # ---- Retry & pacing ----
    def _cooldown(self):
        diff = time.time() - self._last_write_ts
        if diff < self.cfg.write_cooldown_sec:
            time.sleep(self.cfg.write_cooldown_sec - diff)
        self._last_write_ts = time.time()

    def _retry(self, key: str, fn, *a, **kw):
        last = None
        for i in range(META_RETRY_MAX):
            try:
                return fn(*a, **kw)
            except Exception as e:
                last = e
                if i < META_RETRY_MAX - 1:
                    time.sleep(min(META_BACKOFF_BASE * (2 ** i), 8.0))
                else:
                    raise last

    # ---- HTTP helpers ----
    def _graph_abs(self, path: str, method: str = "get", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        a = self.account
        url = f"https://graph.facebook.com/{a.api_version}/{path.lstrip('/')}"
        q = {}
        if params:
            # encode parameters per key
            for k, v in params.items():
                enc = _encode_param_value(k, v)
                if enc is not None:
                    q[k] = enc
        # always include token and proof
        q["access_token"] = a.access_token
        # appsecret_proof recommended for security
        if a.app_secret:
            q["appsecret_proof"] = _appsecret_proof(a.app_secret, a.access_token)

        if method.lower() == "get":
            r = requests.get(url, params=q, timeout=META_TIMEOUT)
        else:
            # For POST edges (like async insights start), Graph expects form-like payload; JSON also works for most.
            r = requests.post(url, data=q if method.lower() == "post" else None, timeout=META_TIMEOUT)

        if r.status_code >= 400:
            raise RuntimeError(f"Graph {method.upper()} {path}: {r.status_code} {r.text}")
        return r.json()

    def _graph(self, endpoint: str, method: str = "get", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Account-scoped path helper: /{version}/{act_id}/{endpoint}
        return self._graph_abs(f"{self.act_id}/{endpoint.lstrip('/')}", method, params)

    # ------------------ INSIGHTS (SYNC) ------------------
    def get_insights(
        self,
        object_id: str,
        *,
        level: Optional[str] = None,
        fields: Optional[List[str]] = None,
        date_preset: Optional[str] = None,
        time_range: Optional[Dict[str, str]] = None,  # {"since":"YYYY-MM-DD","until":"YYYY-MM-DD"}
        since: Optional[str] = None,
        until: Optional[str] = None,
        breakdowns: Optional[List[str]] = None,
        action_breakdowns: Optional[List[str]] = None,
        action_attribution_windows: Optional[List[str]] = None,  # e.g., ["1d_click","7d_click","1d_view"]
        filtering: Optional[List[Dict[str, Any]]] = None,
        sort: Optional[str] = None,  # e.g., "reach_descending"
        time_increment: Union[int, str] = 1,  # or "all_days"
        limit: int = 500,
        paginate: bool = True,
        after: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generic insights fetcher for any object (account/campaign/adset/ad).
        """
        params: Dict[str, Any] = {
            "time_increment": time_increment,
            "limit": limit,
        }
        if level:
            params["level"] = level
        if fields:
            params["fields"] = fields
        if date_preset:
            params["date_preset"] = date_preset
        if sort:
            params["sort"] = sort
        if breakdowns:
            params["breakdowns"] = breakdowns
        if action_breakdowns:
            params["action_breakdowns"] = action_breakdowns
        if action_attribution_windows:
            params["action_attribution_windows"] = action_attribution_windows
        if filtering:
            params["filtering"] = filtering

        # time range handling
        if time_range:
            params["time_range"] = time_range
        elif since and until:
            params["time_range"] = {"since": since, "until": until}

        # initial call
        rows: List[Dict[str, Any]] = []
        call_params = dict(params)
        if after:
            call_params["after"] = after

        res = self._graph_abs(f"{object_id}/insights", "get", call_params)
        rows.extend(res.get("data") or [])

        # pagination
        while paginate and res.get("paging", {}).get("next"):
            curs = res.get("paging", {}).get("cursors", {}).get("after")
            if not curs:
                break
            call_params["after"] = curs
            res = self._graph_abs(f"{object_id}/insights", "get", call_params)
            rows.extend(res.get("data") or [])
        return rows

    def get_ad_insights(
        self,
        *,
        level: str = "ad",
        fields: Optional[List[str]] = None,
        date_preset: Optional[str] = None,
        time_range: Optional[Dict[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        breakdowns: Optional[List[str]] = None,
        action_breakdowns: Optional[List[str]] = None,
        action_attribution_windows: Optional[List[str]] = None,
        filtering: Optional[List[Dict[str, Any]]] = None,
        sort: Optional[str] = None,
        time_increment: Union[int, str] = 1,
        limit: int = 500,
        paginate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convenience for account-level insights (act_<ID>/insights).
        """
        return self.get_insights(
            self.act_id,
            level=level,
            fields=fields or ["spend"],
            date_preset=date_preset,
            time_range=time_range,
            since=since,
            until=until,
            breakdowns=breakdowns,
            action_breakdowns=action_breakdowns,
            action_attribution_windows=action_attribution_windows,
            filtering=filtering,
            sort=sort,
            time_increment=time_increment,
            limit=limit,
            paginate=paginate,
        )

    # ------------------ INSIGHTS (ASYNC) ------------------
    def start_insights_job(
        self,
        object_id: str,
        *,
        level: Optional[str] = None,
        fields: Optional[List[str]] = None,
        date_preset: Optional[str] = None,
        time_range: Optional[Dict[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        breakdowns: Optional[List[str]] = None,
        action_breakdowns: Optional[List[str]] = None,
        action_attribution_windows: Optional[List[str]] = None,
        filtering: Optional[List[Dict[str, Any]]] = None,
        sort: Optional[str] = None,
        time_increment: Union[int, str] = 1,
    ) -> str:
        """
        Kick off an async insights job. Returns job_id.
        """
        params: Dict[str, Any] = {
            "time_increment": time_increment,
            "async": "true",
        }
        if level:
            params["level"] = level
        if fields:
            params["fields"] = fields
        if date_preset:
            params["date_preset"] = date_preset
        if breakdowns:
            params["breakdowns"] = breakdowns
        if action_breakdowns:
            params["action_breakdowns"] = action_breakdowns
        if action_attribution_windows:
            params["action_attribution_windows"] = action_attribution_windows
        if filtering:
            params["filtering"] = filtering
        if sort:
            params["sort"] = sort

        if time_range:
            params["time_range"] = time_range
        elif since and until:
            params["time_range"] = {"since": since, "until": until}

        res = self._graph_abs(f"{object_id}/insights", "post", params)
        job_id = res.get("report_run_id") or res.get("id")
        if not job_id:
            raise RuntimeError(f"Async insights start failed: {res}")
        return str(job_id)

    def poll_insights_job(self, job_id: str) -> Dict[str, Any]:
        """
        Poll async insights job status.
        Returns: {"id":..., "async_status":"Job Completed"|"Job Running"|...}
        """
        return self._graph_abs(f"{job_id}", "get", {"fields": "id,async_status,async_percent_completion"})

    def fetch_insights_job_result(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Once a job is completed, fetch results via /{job_id}/insights.
        """
        rows: List[Dict[str, Any]] = []
        params: Dict[str, Any] = {}
        res = self._graph_abs(f"{job_id}/insights", "get", params)
        rows.extend(res.get("data") or [])
        while res.get("paging", {}).get("next"):
            curs = res.get("paging", {}).get("cursors", {}).get("after")
            if not curs:
                break
            params["after"] = curs
            res = self._graph_abs(f"{job_id}/insights", "get", params)
            rows.extend(res.get("data") or [])
        return rows

    # ------------------ WRITE: Adsets ------------------
    def ensure_adset(
        self,
        campaign_id: str,
        name: str,
        daily_budget: float,
        *,
        optimization_goal: str = "OFFSITE_CONVERSIONS",
        billing_event: str = "IMPRESSIONS",
        bid_strategy: str = "LOWEST_COST_WITHOUT_CAP",
        targeting: Optional[Dict[str, Any]] = None,
        placements: Optional[List[str]] = None,  # ["facebook","instagram"]
        status: str = "PAUSED",
    ) -> Dict[str, Any]:
        targeting = targeting or {"age_min": 18, "geo_locations": {"countries": ["US"]}}
        if placements:
            targeting["publisher_platforms"] = placements
            targeting["facebook_positions"] = ["feed"] if "facebook" in placements else []
            targeting["instagram_positions"] = ["feed", "story", "reels"] if "instagram" in placements else []
        budget_cents = int(max(self.cfg.budget_min, min(self.cfg.budget_max, daily_budget)) * 100)
        params = dict(
            name=name,
            campaign_id=campaign_id,
            daily_budget=budget_cents,
            billing_event=billing_event,
            optimization_goal=optimization_goal,
            bid_strategy=bid_strategy,
            targeting=_sanitize(targeting),
            status=status,
        )
        if self.dry_run:
            return {"mock": True, **params}
        self._cooldown()
        try:
            return self._graph("adsets", "post", params)
        except Exception:
            if USE_SDK:
                self._init_sdk()
                return self._retry(
                    "create_adset",
                    lambda: AdAccount(self.act_id).create_ad_set(fields=[], params=params),
                )
            raise

    # ------------------ WRITE: Creatives ------------------
    def _get_video_thumbnail_url(self, vid: str) -> Optional[str]:
        try:
            # video ids live at top level; hop out
            res = self._graph_abs(f"{vid}/thumbnails", "get", {"fields": "uri,is_preferred"})
            data = res.get("data") or []
            for d in data:
                if d.get("is_preferred"):
                    return d.get("uri")
            return (data[0].get("uri") if data else None)
        except Exception:
            return None

    def create_video_creative(
        self,
        page_id: str,
        name: str,
        *,
        video_library_id: str,
        primary_text: str,
        headline: str,
        description: str = "",
        call_to_action: str = "SHOP_NOW",
        link_url: Optional[str] = None,
        utm_params: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        instagram_actor_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        bad = _contains_forbidden([primary_text, headline, description])
        if bad:
            raise ValueError(f"Forbidden term: {bad}")

        vid_id = (video_library_id or "").strip()
        thumb = thumbnail_url or self._get_video_thumbnail_url(vid_id)
        ig_id = instagram_actor_id or os.getenv("IG_ACTOR_ID")
        link = f"{link_url}?{utm_params}" if (link_url and utm_params) else link_url

        video_data: Dict[str, Any] = {"message": primary_text, "video_id": vid_id}
        if headline:
            video_data["link_description"] = headline[:100]
        if link:
            video_data["call_to_action"] = {"type": call_to_action, "value": {"link": link}}
        if thumb:
            video_data["image_url"] = thumb

        story_spec: Dict[str, Any] = {"page_id": page_id, "video_data": video_data}
        if ig_id:
            story_spec["instagram_actor_id"] = ig_id

        params = {"name": name, "object_story_spec": story_spec}

        if self.dry_run:
            return {"mock": True, "name": name, "video_id": vid_id}

        self._cooldown()
        try:
            return self._graph("adcreatives", "post", params)
        except Exception:
            if USE_SDK:
                self._init_sdk()
                return self._retry(
                    "create_creative",
                    lambda: AdAccount(self.act_id).create_ad_creative(fields=[], params=params),
                )
            raise

    # ------------------ WRITE: Ads ------------------
    def create_ad(self, adset_id: str, name: str, creative_id: str, status: str = "PAUSED") -> Dict[str, Any]:
        payload = {"name": name, "adset_id": adset_id, "creative": {"creative_id": creative_id}, "status": status}
        if self.dry_run:
            return {"mock": True, "adset_id": adset_id, "name": name}
        self._cooldown()
        try:
            return self._graph("ads", "post", payload)
        except Exception:
            if USE_SDK:
                self._init_sdk()
                return self._retry("create_ad", lambda: AdAccount(self.act_id).create_ad(fields=[], params=payload))
            raise

    # ------------------ Convenience helpers ------------------
    def create_validation_adset(self, campaign_id: str, label: str, daily_budget: float = 40.0, *, placements=None):
        return self.ensure_adset(campaign_id, f"[VALID] {label}", daily_budget, placements=placements)

    def create_scaling_adset(self, campaign_id: str, label: str, daily_budget: float = 100.0, *, placements=None):
        return self.ensure_adset(campaign_id, f"[SCALE] {label}", daily_budget, placements=placements)


__all__ = ["AccountAuth", "ClientConfig", "MetaClient"]
