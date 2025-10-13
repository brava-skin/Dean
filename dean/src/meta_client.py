# meta_client.py
from __future__ import annotations

import hashlib, os, random, re, time, requests
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

USE_SDK = False
try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.adobjects.adset import AdSet
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.campaign import Campaign
    from facebook_business.adobjects.ad import Ad
    from facebook_business.exceptions import FacebookRequestError
    USE_SDK = True
except Exception:
    pass

# ------------------------- CONFIG -------------------------
META_TIMEOUT = float(os.getenv("META_TIMEOUT", "30"))
META_RETRY_MAX = int(os.getenv("META_RETRY_MAX", "4"))
META_BACKOFF_BASE = float(os.getenv("META_BACKOFF_BASE", "0.4"))
META_WRITE_COOLDOWN_SEC = int(os.getenv("META_WRITE_COOLDOWN_SEC", "5"))
CB_FAILS = int(os.getenv("META_CB_FAILS", "5"))
CB_RESET_SEC = int(os.getenv("META_CB_RESET_SEC", "120"))
BUDGET_MIN = float(os.getenv("BUDGET_MIN", "5"))
BUDGET_MAX = float(os.getenv("BUDGET_MAX", "50000"))
BUDGET_STEP_PCT = float(os.getenv("BUDGET_MAX_STEP_PCT", "200"))
ACCOUNT_TZ = os.getenv("ACCOUNT_TZ", "Europe/Amsterdam")
ACCOUNT_CCY = os.getenv("ACCOUNT_CURRENCY", "EUR")
HUMAN_CONFIRM_JUMP_PCT = float(os.getenv("HUMAN_CONFIRM_JUMP_PCT", "200"))
FORBIDDEN_TERMS = tuple(t.strip().lower() for t in os.getenv("FORBIDDEN_TERMS", "cures,miracle,guaranteed").split(","))

# ------------------------- HELPERS -------------------------
def _s(x): return str(x) if x not in (None, True, False) else ""
def _sanitize(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None: return obj
    if isinstance(obj, dict): return {str(k): _sanitize(v) for k,v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)): return [_sanitize(v) for v in obj]
    return str(obj)
def _contains_forbidden(texts): 
    for t in texts: 
        for bad in FORBIDDEN_TERMS: 
            if bad and bad in (t or "").lower(): return bad
def _normalize_account_id(account_id: str):
    aid = account_id.strip()
    num = aid[4:] if aid.startswith("act_") else aid
    return num, f"act_{num}"

# ------------------------- DATACLASSES -------------------------
@dataclass
class AccountAuth:
    account_id: str
    access_token: str
    app_id: str
    app_secret: str
    api_version: str = "v23.0"

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
    """Optimized Meta Marketing API client with retry, pacing & compliance guards."""

    def __init__(self, accounts: Union[List[AccountAuth], AccountAuth], cfg: Optional[ClientConfig] = None, *, dry_run=True):
        self.cfg = cfg or ClientConfig()
        self.dry_run = dry_run
        self.accounts = accounts if isinstance(accounts, list) else [accounts]
        self._acct_num, self._acct_act = {}, {}
        for i, acc in enumerate(self.accounts):
            num, act = _normalize_account_id(acc.account_id)
            self._acct_num[i], self._acct_act[i] = num, act
        self._active = 0
        self._fail_count, self._cb_until = {}, {}
        self._last_write_ts = 0
        self._init_sdk()

    # ---- Core props ----
    @property
    def account(self): return self.accounts[self._active]
    @property
    def act_id(self): return self._acct_act[self._active]
    def _init_sdk(self):
        if USE_SDK and not self.dry_run:
            a = self.account
            FacebookAdsApi.init(a.app_id, a.app_secret, a.access_token, api_version=a.api_version)

    # ---- Retry & pacing ----
    def _cooldown(self):
        diff = time.time() - self._last_write_ts
        if diff < self.cfg.write_cooldown_sec: time.sleep(self.cfg.write_cooldown_sec - diff)
        self._last_write_ts = time.time()

    def _retry(self, key, fn, *a, **kw):
        if self._cb_until.get(key, 0) > time.time():
            raise RuntimeError(f"Circuit open for {key}")
        for i in range(META_RETRY_MAX):
            try: return fn(*a, **kw)
            except Exception as e:
                if i < META_RETRY_MAX - 1:
                    time.sleep(min(META_BACKOFF_BASE * 2 ** i, 8))
                else: raise e

    # ---- HTTP helpers ----
    def _graph(self, endpoint, method="get", data=None):
        url = f"https://graph.facebook.com/{self.account.api_version}/{self.act_id}/{endpoint.lstrip('/')}"
        data = _sanitize(data or {})
        data["access_token"] = self.account.access_token
        r = requests.request(method, url, json=data if method=="post" else None, timeout=META_TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError(f"Graph {method.upper()} {endpoint}: {r.status_code} {r.text}")
        return r.json()

    # ---- Adset ensure ----
    def ensure_adset(
        self,
        campaign_id: str,
        name: str,
        daily_budget: float,
        *,
        optimization_goal="OFFSITE_CONVERSIONS",
        billing_event="IMPRESSIONS",
        bid_strategy="LOWEST_COST_WITHOUT_CAP",
        targeting: Optional[Dict[str, Any]] = None,
        placements: Optional[List[str]] = None,
        status="PAUSED",
    ) -> Dict[str, Any]:
        targeting = targeting or {"age_min":18,"geo_locations":{"countries":["US"]}}
        if placements:
            targeting["publisher_platforms"] = placements
            targeting["facebook_positions"] = [p for p in placements if p.lower() in ("feed","marketplace","video_feeds")]
            targeting["instagram_positions"] = [p for p in placements if p.lower() in ("story","feed","reels")]
        budget_cents = int(max(self.cfg.budget_min, min(self.cfg.budget_max, daily_budget)) * 100)
        params = dict(
            name=name, campaign_id=campaign_id, daily_budget=budget_cents,
            billing_event=billing_event, optimization_goal=optimization_goal,
            bid_strategy=bid_strategy, targeting=_sanitize(targeting), status=status
        )
        if self.dry_run: return {"mock":True,**params}
        self._cooldown()
        try:
            return self._graph("adsets","post",params)
        except Exception:
            self._init_sdk()
            return self._retry("create_adset", lambda: AdAccount(self.act_id).create_ad_set(fields=[],params=params))

    # ---- Creative ----
    def _get_video_thumbnail_url(self, vid: str):
        try:
            res = self._graph(f"../../{vid}/thumbnails", "get", {"fields":"uri,is_preferred"})
            data = res.get("data") or []
            for d in data:
                if d.get("is_preferred"): return d.get("uri")
            return data[0]["uri"] if data else None
        except Exception: return None

    def create_video_creative(
        self, page_id: str, name: str, *, video_library_id: str,
        primary_text: str, headline: str, description="", call_to_action="SHOP_NOW",
        link_url=None, utm_params=None, thumbnail_url=None, instagram_actor_id=None
    ):
        bad = _contains_forbidden([primary_text, headline, description])
        if bad: raise ValueError(f"Forbidden term: {bad}")
        vid_id = video_library_id.strip()
        thumb = thumbnail_url or self._get_video_thumbnail_url(vid_id)
        ig_id = instagram_actor_id or os.getenv("IG_ACTOR_ID")
        link = f"{link_url}?{utm_params}" if utm_params else link_url

        video_data = {"message":primary_text,"video_id":vid_id}
        if headline: video_data["link_description"] = headline[:100]
        if link: video_data["call_to_action"] = {"type":call_to_action,"value":{"link":link}}
        if thumb: video_data["image_url"] = thumb

        story_spec = {"page_id":page_id,"video_data":video_data}
        if ig_id: story_spec["instagram_actor_id"] = ig_id
        params = {"name":name,"object_story_spec":story_spec}
        if self.dry_run: return {"mock":True,"name":name,"video_id":vid_id}
        self._cooldown()
        try:
            return self._graph("adcreatives","post",params)
        except Exception:
            self._init_sdk()
            return self._retry("create_creative", lambda: AdAccount(self.act_id).create_ad_creative(fields=[],params=params))

    # ---- Ads ----
    def create_ad(self, adset_id: str, name: str, creative_id: str, status="PAUSED"):
        payload = {"name":name,"adset_id":adset_id,"creative":{"creative_id":creative_id},"status":status}
        if self.dry_run: return {"mock":True,"adset_id":adset_id,"name":name}
        self._cooldown()
        try: return self._graph("ads","post",payload)
        except Exception:
            self._init_sdk()
            return self._retry("create_ad", lambda: AdAccount(self.act_id).create_ad(fields=[],params=payload))

    # ---- Convenience ----
    def create_validation_adset(self, campaign_id:str, label:str, daily_budget=40.0, *, placements=None):
        return self.ensure_adset(campaign_id, f"[VALID] {label}", daily_budget, placements=placements)

    def create_scaling_adset(self, campaign_id:str, label:str, daily_budget=100.0, *, placements=None):
        return self.ensure_adset(campaign_id, f"[SCALE] {label}", daily_budget, placements=placements)

__all__ = ["AccountAuth","ClientConfig","MetaClient"]
