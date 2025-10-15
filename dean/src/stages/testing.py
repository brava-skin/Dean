# testing.py
from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
from zoneinfo import ZoneInfo  # account-local timezone

from slack import alert_kill, alert_promote, notify

UTC = timezone.utc
LOCAL_TZ = ZoneInfo(os.getenv("ACCOUNT_TZ", os.getenv("ACCOUNT_TIMEZONE", "Europe/Amsterdam")))
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "EUR")
ACCOUNT_CURRENCY_SYMBOL = os.getenv("ACCOUNT_CURRENCY_SYMBOL", "€")


# ------------------------- config helpers -------------------------

def _getenv_f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _getenv_i(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _cfg(settings: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = settings
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _cfg_or_env_f(settings: Dict[str, Any], path: str, env: str, default: float) -> float:
    v = _cfg(settings, path, None)
    if v is None:
        return _getenv_f(env, default)
    try:
        return float(v)
    except Exception:
        return default


def _cfg_or_env_i(settings: Dict[str, Any], path: str, env: str, default: int) -> int:
    v = _cfg(settings, path, None)
    if v is None:
        return _getenv_i(env, default)
    try:
        return int(v)
    except Exception:
        return default


def _cfg_or_env_b(settings: Dict[str, Any], path: str, env: str, default: bool) -> bool:
    v = _cfg(settings, path, None)
    if v is None:
        raw = os.getenv(env, str(int(default))).lower()
        return raw in ("1", "true", "yes", "y")
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y")


def _cfg_or_env_list(settings: Dict[str, Any], path: str, env: str, default: List[str]) -> List[str]:
    v = _cfg(settings, path, None)
    if v is None:
        raw = os.getenv(env, ",".join(default))
        return [s.strip() for s in (raw or "").split(",") if s.strip()]
    if isinstance(v, (list, tuple)):
        return list(v)
    return [str(v).strip()]


# ------------------------- small helpers -------------------------

def _today_str() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")


def _daily_key(stage: str, metric: str) -> str:
    return f"daily::{_today_str()}::{stage}::{metric}"


def _ad_day_flag_key(ad_id: str, flag: str) -> str:
    return f"ad::{ad_id}::{flag}::{_today_str()}"


def _now_minute_key(prefix: str) -> str:
    return f"{prefix}::{datetime.now(LOCAL_TZ).strftime('%Y-%m-%dT%H:%M')}"


def _safe_f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v).replace(",", "").strip())
    except Exception:
        return default


def _ctr(row: Dict[str, Any]) -> float:
    imps = _safe_f(row.get("impressions"))
    clicks = _safe_f(row.get("clicks"))
    return (clicks / imps) if imps > 0 else 0.0


def _roas(row: Dict[str, Any]) -> float:
    roas_list = row.get("purchase_roas") or []
    try:
        if roas_list:
            return float(roas_list[0].get("value", 0)) or 0.0
    except Exception:
        pass
    return 0.0


def _purchase_and_atc_counts(row: Dict[str, Any]) -> Tuple[int, int]:
    acts = row.get("actions") or []
    purch = 0
    atc = 0
    for a in acts:
        t = a.get("action_type")
        v = _safe_f(a.get("value"), 0.0)
        if t == "purchase":
            purch += int(v)
        elif t == "add_to_cart":
            atc += int(v)
    return purch, atc


def _meets_minimums(row: Dict[str, Any], min_impressions: int, min_clicks: int, min_spend: float) -> bool:
    return (
        _safe_f(row.get("spend")) >= min_spend
        and _safe_f(row.get("impressions")) >= min_impressions
        and _safe_f(row.get("clicks")) >= min_clicks
    )


def _stable_pass(store: Any, entity_id: str, rule_key: str, condition: bool, consec_required: int) -> bool:
    key = f"{entity_id}::stable::{rule_key}"
    if condition:
        try:
            val = store.incr(key, 1)
        except Exception:
            val = 1
            try:
                store.set_counter(key, val)
            except Exception:
                pass
        return val >= consec_required
    try:
        store.set_counter(key, 0)
    except Exception:
        pass
    return False


def _active_count(ads_list: List[Dict[str, Any]]) -> int:
    return sum(1 for a in ads_list if str(a.get("status", "")).upper() == "ACTIVE")


def _get_counter_float(store: Any, key: str, scale: int = 1_000_000) -> Optional[float]:
    try:
        v = store.get_counter(key)
        return None if v == 0 else float(v) / scale
    except Exception:
        return None


def _set_counter_float(store: Any, key: str, val: float, scale: int = 1_000_000) -> None:
    try:
        store.set_counter(key, int(round(val * scale)))
    except Exception:
        pass


def _update_ewma_ctr(store: Any, ad_id: str, ctr_today: float, ewma_alpha: float) -> float:
    k = f"{ad_id}::ewma_ctr"
    prev = _get_counter_float(store, k) or ctr_today
    ewma = ewma_alpha * ctr_today + (1.0 - ewma_alpha) * prev
    _set_counter_float(store, k, ewma)
    return ewma


def _fatigue_detect(store: Any, ad_id: str, ctr_today: float, ewma_alpha: float, drop_pct: float) -> bool:
    ewma = _update_ewma_ctr(store, ad_id, ctr_today, ewma_alpha)
    return ewma > 0 and ctr_today < (1.0 - drop_pct) * ewma


def _bayes_kill_prob(clicks: float, imps: float, floor: float, prior_a: float, prior_b: float, sample_count: int) -> float:
    a = prior_a + max(0.0, clicks)
    b = prior_b + max(0.0, imps - clicks)
    below = 0
    n = max(1, sample_count)
    for _ in range(n):
        sample = random.betavariate(a, b)
        if sample < floor:
            below += 1
    return below / n


def _adaptive_ctr_floor(
    rows: List[Dict[str, Any]],
    min_impressions_for_floor: int,
    floor_min: float,
    floor_scale: float,
) -> float:
    ctrs = sorted((_ctr(r) for r in rows if _safe_f(r.get("impressions")) >= min_impressions_for_floor), reverse=True)
    if not ctrs:
        return floor_min
    mid = ctrs[len(ctrs) // 2]
    return max(floor_min, mid * floor_scale)


def _data_quality_sentry(row: Dict[str, Any], min_spend_for_alert: float) -> Optional[str]:
    spend = _safe_f(row.get("spend"))
    purch, atc = _purchase_and_atc_counts(row)
    if spend >= min_spend_for_alert and purch == 0 and atc == 0:
        return "Spend present but no actions, check tracking"
    return None


def _bandit_score_for_queue_item(store: Any, label: str, prior_a: float, prior_b: float) -> float:
    try:
        clicks = store.get_counter(f"qctr::{label}::clicks")
        imps = store.get_counter(f"qctr::{label}::imps")
    except Exception:
        clicks, imps = 0, 0
    a = prior_a + clicks
    b = prior_b + max(0, imps - clicks)
    return random.betavariate(a, b)


def _record_queue_feedback(store: Any, label: str, clicks: float, imps: float) -> None:
    try:
        store.incr(f"qctr::{label}::clicks", int(clicks))
        store.incr(f"qctr::{label}::imps", int(imps))
    except Exception:
        pass


def _clean_text_token(v: Any) -> str:
    s = str(v or "")
    s = s.replace("_", " ")
    return " ".join(s.split()).strip()


def _label_from_row(row: Dict[str, Any]) -> str:
    avatar = _clean_text_token(row.get("avatar"))
    vis = _clean_text_token(row.get("visual_style"))
    script = _clean_text_token(row.get("script"))
    if avatar and vis and script:
        return f"{avatar} - {vis} - {script}"
    fname = str(row.get("filename") or "").strip()
    if fname:
        base = os.path.basename(fname)
        stem, _ = os.path.splitext(base)
        parts = [p for p in stem.split("_") if p]
        if len(parts) >= 3:
            a, v, s = parts[-3], parts[-2], parts[-1]
            return f"{_clean_text_token(a)} - {_clean_text_token(v)} - {_clean_text_token(s)}"
        if stem:
            return _clean_text_token(stem)
    return "UNNAMED"


def _load_copy_bank(settings: Dict[str, Any]) -> Dict[str, Any]:
    path = (settings.get("copy_bank") or {}).get("path")
    if not path:
        return {"global": {}}
    p = Path(path)
    if not p.exists():
        notify(f"⚠️ [TEST] Copy bank not found at '{p}'.")
        return {"global": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        notify(f"⚠️ [TEST] Failed to parse copy bank JSON: {e}")
        return {"global": {}}


def _rr_index(store: Any, key: str, n: int) -> int:
    n = max(1, n)
    try:
        i = store.incr(f"rr::{key}", 1) - 1
        return i % n
    except Exception:
        return random.randrange(n)


def _choose_mix_and_match(store: Any, copy_bank: Dict[str, Any], label_key: str, strategy: str = "round_robin") -> Tuple[str, str, str]:
    g = (copy_bank.get("global") or {})
    pts = g.get("primary_texts") or []
    hls = g.get("headlines") or []
    descs = g.get("descriptions") or []
    if not (pts and hls and descs):
        raise ValueError("Copy bank needs non-empty arrays: primary_texts, headlines, descriptions.")
    def pick(items: List[str], subkey: str) -> str:
        if strategy == "round_robin":
            return items[_rr_index(store, f"mix::{label_key}::{subkey}", len(items))]
        return random.choice(items)
    return pick(pts, "pt"), pick(hls, "hl"), pick(descs, "desc")


# -------- robust normalizer for Excel-mangled video IDs --------
def _normalize_video_id_cell(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().strip("'").strip('"')
    if s == "" or s.lower() in ("nan", "none", "null"):
        return ""
    s = s.replace(",", "").replace(" ", "")
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d+(\.\d+)?[eE][+-]?\d+", s):
        try:
            from decimal import Decimal, getcontext
            getcontext().prec = 50
            return str(int(Decimal(s)))
        except Exception:
            return ""
    if re.fullmatch(r"\d+", s):
        return s
    keep = "".join(ch for ch in s if ch.isdigit())
    return keep


# ---------- pause/resume helpers and one-shot alerts ----------

def _pause_ad(meta: Any, ad_id: str) -> Any:
    if hasattr(meta, "pause_ad") and callable(getattr(meta, "pause_ad")):
        return meta.pause_ad(ad_id)
    try:
        import requests
        ver = getattr(getattr(meta, "account", None), "api_version", "v23.0") or "v23.0"
        token = getattr(getattr(meta, "account", None), "access_token", None)
        if not token:
            raise RuntimeError("No access token on meta client.")
        url = f"https://graph.facebook.com/{ver}/{ad_id}"
        r = requests.post(url, json={"access_token": token, "status": "PAUSED"}, timeout=30)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"error": {"message": r.text}}
            raise RuntimeError(f"Graph POST {url} {r.status_code}: {err}")
        return r.json()
    except Exception:
        try:
            from facebook_business.adobjects.ad import Ad as FBAd
            FBAd(ad_id).api_update(params={"status": "PAUSED"})
            return {"result": "ok", "sdk": True}
        except Exception as e2:
            raise e2


def _paused_alerted_key(ad_id: str) -> str:
    return f"ad::{ad_id}::paused_alerted"


def _set_paused_alerted(store: Any, ad_id: str, val: int) -> None:
    try:
        store.set_counter(_paused_alerted_key(ad_id), int(val))
    except Exception:
        pass


def _get_paused_alerted(store: Any, ad_id: str) -> int:
    try:
        return int(store.get_counter(_paused_alerted_key(ad_id)))
    except Exception:
        return 0


# ---------- helpers for promotion cloning ----------

def _get_ad_creative_id(meta: Any, ad_id: str) -> Optional[str]:
    """
    Return the creative id of an ad. Uses client method if present, else Graph fallback, else SDK.
    """
    # client method
    if hasattr(meta, "get_ad_details") and callable(getattr(meta, "get_ad_details")):
        try:
            d = meta.get_ad_details(ad_id)
            cid = (d.get("creative") or {}).get("id")
            if cid:
                return str(cid)
        except Exception:
            pass
    # Graph fallback
    try:
        import requests
        ver = getattr(getattr(meta, "account", None), "api_version", "v23.0") or "v23.0"
        token = getattr(getattr(meta, "account", None), "access_token", None)
        if token:
            url = f"https://graph.facebook.com/{ver}/{ad_id}"
            r = requests.get(url, params={"access_token": token, "fields": "creative{id}"}, timeout=30)
            j = r.json()
            cid = (j.get("creative") or {}).get("id")
            if cid:
                return str(cid)
    except Exception:
        pass
    # SDK fallback
    try:
        from facebook_business.adobjects.ad import Ad as FBAd
        ad = FBAd(ad_id)
        fields = ["creative"]
        ad.remote_read(fields=fields)
        cid = (ad.get("creative") or {}).get("id")
        if cid:
            return str(cid)
    except Exception:
        pass
    return None


# ---------- IG helpers (new) ----------

def _resolve_page_instagram_ids(page_id: str, token: Optional[str], ver: str = "v23.0") -> Dict[str, Optional[str]]:
    """
    Returns {'user_id': <connected/page IG id>, 'business_id': <business ig id>, 'source': 'connected'|'business'|None}
    """
    out = {"user_id": None, "business_id": None, "source": None}
    if not (page_id and token):
        return out
    try:
        import requests
        url = f"https://graph.facebook.com/{ver}/{page_id}"
        params = {
            "access_token": token,
            "fields": "instagram_business_account{id,username},connected_instagram_account{id,username}",
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        biz = (data.get("instagram_business_account") or {}).get("id")
        con = (data.get("connected_instagram_account") or {}).get("id")
        # Prefer page-connected (instagram_user_id) for creatives, else business id.
        if con:
            out["user_id"] = str(con)
            out["source"] = "connected"
        if biz:
            out["business_id"] = str(biz)
            if not out["user_id"]:
                out["user_id"] = str(biz)
                out["source"] = "business"
        return out
    except Exception:
        return out


def _get_ad_account_id(meta: Any) -> Optional[str]:
    """Return ad account id without 'act_' prefix if possible."""
    try:
        acct = getattr(meta, "account", None)
        if not acct:
            return None
        cand = getattr(acct, "ad_account_id", None) or getattr(acct, "id", None) or getattr(acct, "account_id", None)
        if not cand:
            return None
        s = str(cand)
        return s[4:] if s.startswith("act_") else s
    except Exception:
        return None


def _resolve_video_thumbnail_url(token: Optional[str], ver: str, video_id: str) -> Optional[str]:
    """Return a usable thumbnail URL for a video_id, or None."""
    if not (token and video_id):
        return None
    try:
        import requests
        url = f"https://graph.facebook.com/{ver}/{video_id}/thumbnails"
        params = {"access_token": token, "fields": "uri,is_preferred", "limit": 5}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        items = data.get("data") or []
        if not items:
            return None
        pref = next((i for i in items if i.get("is_preferred")), None)
        return (pref or items[0]).get("uri")
    except Exception:
        return None


def _graph_create_video_creative_instagram_user(
    *,
    token: str,
    api_version: str,
    ad_account_id: str,
    page_id: str,
    instagram_user_id: str,
    video_id: str,
    link_url: str,
    message: str = "",
    link_description: str = "",
    thumbnail_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Direct Graph call to create a video creative using instagram_user_id (for page-connected IG).
    Returns {'id': '<creative_id>'} on success; raises on errors.
    """
    import requests

    video_data: Dict[str, Any] = {
        "video_id": str(video_id),
        "call_to_action": {"type": "SHOP_NOW", "value": {"link": link_url}},
    }
    if thumbnail_url:
        video_data["image_url"] = thumbnail_url
    if message:
        video_data["message"] = message
    if link_description:
        video_data["link_description"] = link_description

    spec = {
        "page_id": str(page_id),
        "instagram_user_id": str(instagram_user_id),
        "video_data": video_data,
    }

    url = f"https://graph.facebook.com/{api_version}/act_{ad_account_id}/adcreatives"
    payload = {
        "name": "IGUserCreative",
        "object_story_spec": json.dumps(spec, separators=(",", ":")),
        "access_token": token,
    }
    r = requests.post(url, data=payload, timeout=30)
    try:
        j = r.json()
    except Exception:
        j = {"error": {"message": r.text}}
    if r.status_code >= 400 or "error" in j:
        raise RuntimeError(f"Graph create creative failed ({r.status_code}): {j}")
    return j


# ---------------------------------------------------------------------------

def run_testing_tick(
    meta: Any,
    settings: Dict[str, Any],
    engine: Any,
    store: Any,
    queue_df: pd.DataFrame,
    set_supabase_status: Callable[[List[str], str], None],
    *,
    placements: Optional[List[str]] = None,           # respected if provided
    instagram_actor_id: Optional[str] = None,         # legacy support; we now prefer instagram_user_id
) -> Dict[str, Any]:
    summary = {"kills": 0, "promotions": 0, "launched": 0, "fatigue_flags": 0, "data_quality_alerts": 0}
    try:
        tk = _now_minute_key("testing")
        if hasattr(store, "tick_seen") and store.tick_seen(tk):
            notify(f"ℹ️ [TEST] Tick {tk} already processed; skipping.")
            return summary
    except Exception:
        pass

    # --- Resolve config from YAML with env fallbacks ---
    tmin = (settings.get("testing") or {}).get("minimums") or {}
    min_impressions = int(tmin.get("min_impressions", _getenv_i("TEST_MIN_IMPRESSIONS", 300)))
    min_clicks = int(tmin.get("min_clicks", _getenv_i("TEST_MIN_CLICKS", 10)))
    min_spend = float(tmin.get("min_spend_eur", tmin.get("min_spend", _getenv_f("TEST_MIN_SPEND", 10.0))))

    min_spend_before_kill = _cfg_or_env_f(settings, "testing.engine.fairness.min_spend_before_kill_eur", "TEST_MIN_SPEND_BEFORE_KILL", 20.0)

    consec_required = _cfg_or_env_i(settings, "stability.consecutive_ticks", "TEST_CONSEC_TICKS_REQUIRED", 2)

    lifetime_spend_no_purchase_eur = _cfg_or_env_f(settings, "testing.engine.tripwires.lifetime_spend_no_purchase_eur", "TEST_SPEND_NO_PURCHASE_EUR", 32.0)

    prior_a = _cfg_or_env_f(settings, "testing.engine.bayes.prior_a", "TEST_BETA_PRIOR_A", 2.0)
    prior_b = _cfg_or_env_f(settings, "testing.engine.bayes.prior_b", "TEST_BETA_PRIOR_B", 300.0)
    sample_count = _cfg_or_env_i(settings, "testing.engine.bayes.sample_count", "TEST_BANDIT_SAMPLE_COUNT", 32)
    kill_prob = _cfg_or_env_f(settings, "testing.engine.bayes.kill_prob", "TEST_BAYES_KILL_PROB", 0.90)

    floor_min = _cfg_or_env_f(settings, "testing.engine.adaptive_ctr.floor_min", "TEST_ADAPTIVE_CTR_FLOOR_MIN", 0.008)
    floor_scale = _cfg_or_env_f(settings, "testing.engine.adaptive_ctr.floor_scale", "TEST_ADAPTIVE_CTR_FLOOR_SCALE", 0.70)
    min_imps_for_floor = _cfg_or_env_i(settings, "testing.engine.adaptive_ctr.min_impressions", "TEST_MIN_IMPRESSIONS_FOR_FLOOR", 300)

    ewma_alpha = _cfg_or_env_f(settings, "testing.engine.fatigue.ewma_alpha", "TEST_EWMA_ALPHA", 0.30)
    fatigue_drop_pct = _cfg_or_env_f(settings, "testing.engine.fatigue.drop_pct", "TEST_FATIGUE_DROP_PCT", 0.35)

    pause_after_promotion = _cfg_or_env_b(settings, "testing.engine.promotion.pause_after_promotion", "TEST_PAUSE_AFTER_PROMOTION", True)
    validation_budget_eur = _cfg_or_env_f(settings, "testing.engine.promotion.validation_budget_eur", "TEST_VALIDATION_BUDGET_EUR", 40.0)
    promotion_placements = _cfg_or_env_list(settings, "testing.engine.promotion.placements", "TEST_PROMOTION_PLACEMENTS", ["facebook", "instagram"])

    max_launches_per_tick = _cfg_or_env_i(settings, "testing.engine.queue.max_launches_per_tick", "TEST_MAX_LAUNCHES_PER_TICK", 8)

    attr_windows = _cfg_or_env_list(settings, "testing.engine.attribution_windows", "TEST_ATTR_WINDOWS", ["7d_click", "1d_view"])

    adset_id = _cfg(settings, "ids.testing_adset_id")
    validation_campaign_id = _cfg(settings, "ids.validation_campaign_id")

    keep_live = int(_cfg(settings, "queue_policies.keep_active_ads", _cfg(settings, "testing.keep_ads_live", 4)))

    copy_bank = _load_copy_bank(settings)
    copy_strategy = (settings.get("copy_bank") or {}).get("strategy", "round_robin")

    # -------- Build ACTIVE or PAUSED map --------
    try:
        current_ads = meta.list_ads_in_adset(adset_id)
    except Exception as e:
        notify(f"❗ [TEST] Could not list ads in ad set: {e}")
        current_ads = []

    status_by_ad_id: Dict[str, str] = {}
    for a in current_ads or []:
        try:
            status_by_ad_id[str(a.get("id"))] = str(a.get("status", "")).upper()
        except Exception:
            pass

    for aid, st in status_by_ad_id.items():
        if st == "ACTIVE":
            _set_paused_alerted(store, aid, 0)

    # -------- Fetch insights (today and lifetime) --------
    try:
        rows_today = meta.get_ad_insights(
            level="ad",
            filtering=[{"field": "adset.id", "operator": "IN", "value": [adset_id]}],
            fields=[
                "ad_id",
                "ad_name",
                "adset_id",
                "campaign_id",
                "spend",
                "impressions",
                "clicks",
                "actions",
                "action_values",
                "purchase_roas",
                "reach",
                "unique_clicks",
            ],
            action_attribution_windows=list(attr_windows),
            paginate=True,
            date_preset="today",
        )

        rows_lifetime = meta.get_ad_insights(
            level="ad",
            filtering=[{"field": "adset.id", "operator": "IN", "value": [adset_id]}],
            fields=["ad_id", "ad_name", "spend", "actions", "action_values"],
            action_attribution_windows=list(attr_windows),
            date_preset="lifetime",
            paginate=True,
        )
    except Exception as e:
        notify(f"❗ [TEST] Failed to fetch insights: {e}")
        return summary

    lifetime_by_id: Dict[str, Dict[str, Any]] = {}
    for lr in rows_lifetime or []:
        lifetime_by_id[str(lr.get("ad_id") or "")] = lr

    ctr_floor = _adaptive_ctr_floor(rows_today, min_imps_for_floor, floor_min, floor_scale)

    for r in rows_today:
        ad_id = r.get("ad_id")
        name = r.get("ad_name", "")
        if not ad_id:
            continue

        cur_status = status_by_ad_id.get(str(ad_id), "")
        if cur_status == "PAUSED":
            _set_paused_alerted(store, ad_id, 1)
            continue

        spend_today = _safe_f(r.get("spend"))
        ctr_today = _ctr(r)
        purch_today, atc_today = _purchase_and_atc_counts(r)

        lr = lifetime_by_id.get(str(ad_id), {}) or {}
        spend_life = _safe_f(lr.get("spend"))
        purch_life, atc_life = _purchase_and_atc_counts(lr)

        try:
            notify(f"ℹ️ [TEST] {name}: lifetime_spend={spend_life:.2f} purchases={purch_life} today_spend={spend_today:.2f}")
        except Exception:
            pass

        if (spend_today < min_spend_before_kill and spend_life < min_spend_before_kill) and \
           not _meets_minimums(r, min_impressions, min_clicks, min_spend):
            continue

        dq = _data_quality_sentry(r, min_spend_for_alert=20.0)
        if dq:
            notify(f"🩺 [TEST] {name}: {dq}")
            summary["data_quality_alerts"] += 1

        # fatigue
        if _fatigue_detect(store, ad_id, ctr_today, ewma_alpha=ewma_alpha, drop_pct=fatigue_drop_pct):
            flag_key = _ad_day_flag_key(ad_id, "fatigue")
            first_time_today = False
            try:
                if int(store.get_counter(flag_key) or 0) == 0:
                    store.set_counter(flag_key, 1)
                    first_time_today = True
            except Exception:
                first_time_today = True
                try:
                    store.set_counter(flag_key, 1)
                except Exception:
                    pass
            if first_time_today:
                try:
                    store.incr(_daily_key("TEST", "fatigued"), 1)
                except Exception:
                    pass
            notify(f"🟡 [TEST] Fatigue signal on {name} (CTR drop vs trend).")
            summary["fatigue_flags"] += 1

        bayes_p = _bayes_kill_prob(
            _safe_f(r.get("clicks")),
            _safe_f(r.get("impressions")),
            ctr_floor,
            prior_a=prior_a,
            prior_b=prior_b,
            sample_count=sample_count,
        )
        bayes_kill = bayes_p >= kill_prob
        try:
            kill, reason_engine = engine.should_kill_testing(r)
        except Exception:
            kill, reason_engine = False, ""

        # tripwire kill
        if spend_life >= lifetime_spend_no_purchase_eur and purch_life == 0:
            reason = f"Lifetime spend≥{ACCOUNT_CURRENCY_SYMBOL}{int(lifetime_spend_no_purchase_eur)} & no purchase"
            try:
                _pause_ad(meta, ad_id)

                if _get_paused_alerted(store, ad_id) == 0:
                    try:
                        alert_kill("TEST", name, reason, {"CTR": f"{ctr_today:.2%}", "ROAS": f"{_roas(r):.2f}"})
                    except Exception:
                        notify(f"🛑 [TEST] Killed {name} — {reason}")
                    _set_paused_alerted(store, ad_id, 1)

                try:
                    store.incr(_daily_key("TEST", "kills"), 1)
                except Exception:
                    pass

                store.log(
                    entity_type="ad",
                    entity_id=ad_id,
                    action="PAUSE",
                    reason=f"[TEST] {reason}",
                    level="warn",
                    stage="TEST",
                    rule_type="testing_kill_tripwire",
                    thresholds={"spend_no_purchase_eur": lifetime_spend_no_purchase_eur},
                    observed={
                        "spend_lifetime": spend_life,
                        "purchases_lifetime": purch_life,
                        "CTR_today": f"{ctr_today:.2%}",
                        "ROAS": round(_roas(r), 2),
                    },
                )
                summary["kills"] += 1
            except Exception as e:
                notify(f"❗ [TEST] Failed to pause {name}: {e}")
            continue  # do not consider promotion after a kill

        # other kill paths with stability
        should_kill_other = kill or bayes_kill
        if should_kill_other and _stable_pass(store, ad_id, "kill_test", True, consec_required):
            try:
                _pause_ad(meta, ad_id)
                reason = (
                    reason_engine
                    or ("CTR<{:.2%} (p={:.2f})".format(ctr_floor, bayes_p) if bayes_kill else "Rule-based kill")
                )

                if _get_paused_alerted(store, ad_id) == 0:
                    try:
                        alert_kill("TEST", name, reason, {"CTR": f"{ctr_today:.2%}", "ROAS": f"{_roas(r):.2f}"})
                    except Exception:
                        notify(f"🛑 [TEST] Killed {name} — {reason}")
                    _set_paused_alerted(store, ad_id, 1)

                try:
                    store.incr(_daily_key("TEST", "kills"), 1)
                except Exception:
                    pass

                store.log(
                    entity_type="ad",
                    entity_id=ad_id,
                    action="PAUSE",
                    reason=f"[TEST] {reason}",
                    level="warn",
                    stage="TEST",
                    rule_type="testing_kill",
                    thresholds={"ctr_floor": ctr_floor, "bayes_prob_threshold": kill_prob},
                    observed={
                        "CTR_today": f"{ctr_today:.2%}",
                        "spend_today": spend_today,
                        "ROAS": round(_roas(r), 2),
                    },
                )
                summary["kills"] += 1
            except Exception as e:
                notify(f"❗ [TEST] Failed to pause {name}: {e}")
            continue
        else:
            _stable_pass(store, ad_id, "kill_test", False, consec_required)

        # ---------------- Promotion with actual launch into VALID ----------------
        try:
            adv, adv_reason = engine.should_advance_from_testing(r)
        except Exception:
            adv, adv_reason = False, ""

        if adv and _stable_pass(store, ad_id, "adv_test", True, consec_required):
            label = name.replace("[TEST]", "").strip() or f"Ad_{ad_id}"

            promo_places = list(placements) if placements else list(promotion_placements)

            valid_as = None
            val_ad = None
            try:
                valid_as = meta.create_validation_adset(
                    validation_campaign_id,
                    label,
                    daily_budget=float(validation_budget_eur),
                    placements=promo_places,
                )

                creative_id = _get_ad_creative_id(meta, ad_id)
                if not creative_id:
                    raise RuntimeError("Could not fetch creative id to promote.")

                val_ad_name = f"[VALID] {label}"
                # Use promote_ad_with_continuity to maintain the same ad ID across stages
                if hasattr(meta, "promote_ad_with_continuity"):
                    val_ad = meta.promote_ad_with_continuity(
                        original_ad_id=ad_id,
                        new_adset_id=valid_as["id"],
                        new_name=val_ad_name,
                        creative_id=creative_id,
                        status="ACTIVE",
                    )
                else:
                    # Fallback to regular create_ad if promote_ad_with_continuity is not available
                    val_ad = meta.create_ad(valid_as["id"], val_ad_name, creative_id=creative_id, status="ACTIVE")

                if pause_after_promotion:
                    try:
                        _pause_ad(meta, ad_id)
                        _set_paused_alerted(store, ad_id, 1)
                    except Exception:
                        pass

                try:
                    alert_promote("TEST", "VALID", label, budget=float(validation_budget_eur))
                except Exception:
                    notify(f"✅ [TEST→VALID] {label} — {adv_reason} (Budget: {ACCOUNT_CURRENCY_SYMBOL}{validation_budget_eur:.0f}/day)")

                try:
                    store.incr(_daily_key("TEST", "promotions"), 1)
                except Exception:
                    pass

                # Include ID continuity information in the promotion log
                promotion_meta = {
                    "validation_adset": valid_as, 
                    "placements": promo_places,
                    "id_continuity": True,  # Flag indicating this ad maintains the same ID
                    "original_ad_id": ad_id,
                    "promoted_ad_id": val_ad.get("id") if val_ad else None,
                }
                store.log(
                    entity_type="ad",
                    entity_id=val_ad["id"],
                    action="TEST_TO_VALID",
                    reason=adv_reason,
                    level="info",
                    stage="VALID",
                    meta=promotion_meta,
                )

                try:
                    set_supabase_status([creative_id], "promoted")
                except Exception:
                    pass

                summary["promotions"] += 1

            except Exception as e:
                notify(f"❗ [VALID] Promotion failed for '{label}': {e}")
        else:
            _stable_pass(store, ad_id, "adv_test", False, consec_required)

    # -------- Top-up launches to keep N active --------
    if not current_ads:
        try:
            current_ads = meta.list_ads_in_adset(adset_id)
        except Exception as e:
            notify(f"❗ [TEST] Could not list ads in ad set: {e}")
            return summary

    active_count = _active_count(current_ads)
    need = max(0, min(max_launches_per_tick, keep_live - active_count))
    if need <= 0:
        return summary

    if queue_df is None or queue_df.empty:
        notify(f"⚠️ [TEST] Queue empty; cannot top-up to {keep_live}.")
        return summary

    existing_names = {str(a.get("name", "")).strip() for a in current_ads}
    candidates: List[Dict[str, Any]] = []
    for idx, row in queue_df.iterrows():
        label_core = _label_from_row(row)
        cname = f"[TEST] {label_core}"
        if not label_core or cname in existing_names:
            continue
        d = row.to_dict()
        d["_label_core"] = label_core
        d["_index"] = idx
        candidates.append(d)

    if not candidates:
        notify("ℹ️ [TEST] No eligible queue items (likely duplicates).")
        return summary

    ranked = sorted(
        candidates,
        key=lambda r: _bandit_score_for_queue_item(
            store,
            str(r.get("_label_core") or _label_from_row(r)).strip(),
            prior_a=prior_a,
            prior_b=prior_b,
        ),
        reverse=True,
    )
    picks = ranked[:need]

    for p in picks:
        label_core = str(p.get("_label_core") or _label_from_row(p)).strip()
        cname = f"[TEST] {label_core}"

        video_id_raw = p.get("video_id")
        video_id = _normalize_video_id_cell(video_id_raw)
        if not label_core or not video_id or not video_id.isdigit():
            notify(f"⚠️ [TEST] Skipping '{label_core or 'UNNAMED'}' — invalid video_id (got: {video_id_raw!r}).")
            continue

        try:
            primary_text, headline, description = _choose_mix_and_match(store, _load_copy_bank(settings), label_core, copy_strategy)
        except Exception as e:
            notify(f"⚠️ [TEST] Copy bank issue for '{label_core}': {e}")
            continue

        # Resolve FB Page and IG
        page_id = (p.get("page_id") or os.getenv("FB_PAGE_ID") or "").strip()
        if not page_id:
            notify("⚠️ [TEST] '{label}' missing page_id; set FB_PAGE_ID env or include page_id in the queue row.")
            continue

        token = getattr(getattr(meta, "account", None), "access_token", None)
        ver = getattr(getattr(meta, "account", None), "api_version", "v23.0") or "v23.0"
        resolved = _resolve_page_instagram_ids(page_id, token, ver=ver)
        ig_user_id = resolved.get("user_id")

        ig_actor_env = ((instagram_actor_id or os.getenv("IG_ACTOR_ID") or os.getenv("INSTAGRAM_ACTOR_ID") or "").strip() or None)

        # Thumbnail (video_data requires image_url or image_hash in many cases)
        thumb = (p.get("thumbnail_url") or os.getenv("DEFAULT_THUMB_URL") or "").strip()
        if not thumb:
            auto_thumb = _resolve_video_thumbnail_url(token, ver, str(video_id))
            if auto_thumb:
                thumb = auto_thumb
            else:
                notify("ℹ️ [TEST] Could not auto-resolve video thumbnail; consider setting DEFAULT_THUMB_URL.")

        # Build client kwargs (used when not using instagram_user_id path)
        creative_kwargs: Dict[str, Any] = dict(
            page_id=page_id,
            name=cname,
            video_library_id=video_id,
            primary_text=primary_text,
            headline=headline,
            description=description,
            call_to_action="SHOP_NOW",
            link_url=os.getenv("STORE_URL"),
            utm_params=p.get("utm_params") or None,
            thumbnail_url=thumb or None,
        )

        try:
            if ig_user_id:
                # Graph fallback because client doesn't accept instagram_user_id
                ad_account_id = _get_ad_account_id(meta)
                if not (token and ad_account_id):
                    raise RuntimeError("Missing token or ad_account_id for Graph fallback.")
                link_url = os.getenv("STORE_URL") or ""
                creative = _graph_create_video_creative_instagram_user(
                    token=token,
                    api_version=ver,
                    ad_account_id=str(ad_account_id),
                    page_id=str(page_id),
                    instagram_user_id=str(ig_user_id),
                    video_id=str(video_id),
                    link_url=link_url,
                    message=primary_text or "",
                    link_description=headline or "",
                    thumbnail_url=(thumb or None),
                )
            else:
                if ig_actor_env:
                    creative_kwargs["instagram_actor_id"] = ig_actor_env
                creative = meta.create_video_creative(**creative_kwargs)

            ad = meta.create_ad(adset_id, cname, creative_id=creative["id"], status="ACTIVE")

            try:
                store.incr(_daily_key("TEST", "launched"), 1)
            except Exception:
                pass

            mode = "instagram_user_id" if ig_user_id else ("instagram_actor_id" if ig_actor_env else "page_only")
            store.log(
                entity_type="ad",
                entity_id=ad["id"],
                action="CREATE",
                reason="Top-up to keep_ads_live",
                level="info",
                stage="TEST",
                meta={"testing_adset_id": adset_id, "ig_mode": mode},
            )
            notify(f"🟢 [TEST] Launched {label_core}")
            summary["launched"] += 1
            _record_queue_feedback(store, label_core, clicks=1, imps=200)

            try:
                cid = str(p.get("creative_id") or "").strip()
                if cid:
                    set_supabase_status([cid], "launched")
                else:
                    vid = _normalize_video_id_cell(p.get("video_id"))
                    if vid:
                        set_supabase_status([vid], "launched")
            except Exception:
                pass

        except Exception as e:
            notify(f"❗ [TEST] Failed to launch '{label_core}': {e}")

    return summary
