from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
from zoneinfo import ZoneInfo  # <- use account-local timezone

from slack import alert_kill, alert_promote, notify

UTC = timezone.utc
LOCAL_TZ = ZoneInfo(os.getenv("ACCOUNT_TZ", os.getenv("ACCOUNT_TIMEZONE", "Europe/Amsterdam")))
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "EUR")
ACCOUNT_CURRENCY_SYMBOL = os.getenv("ACCOUNT_CURRENCY_SYMBOL", "€")

# -------- env defaults (used only if YAML keys are missing) --------
_ENV_MIN_IMPRESSIONS = int(os.getenv("TEST_MIN_IMPRESSIONS", "300"))
_ENV_MIN_CLICKS = int(os.getenv("TEST_MIN_CLICKS", "10"))
_ENV_MIN_SPEND = float(os.getenv("TEST_MIN_SPEND", "10"))  # EUR
_CONSEC_TICKS_REQUIRED = int(os.getenv("TEST_CONSEC_TICKS_REQUIRED", "2"))
_ATTR_WINDOWS = tuple((os.getenv("TEST_ATTR_WINDOWS", "7d_click,1d_view") or "7d_click,1d_view").split(","))
_PAUSE_AFTER_PROMOTION = (os.getenv("TEST_PAUSE_AFTER_PROMOTION", "1").lower() in ("1", "true", "yes"))

_BETA_PRIOR_A = float(os.getenv("TEST_BETA_PRIOR_A", "2.0"))
_BETA_PRIOR_B = float(os.getenv("TEST_BETA_PRIOR_B", "300.0"))
_BAYES_KILL_PROB = float(os.getenv("TEST_BAYES_KILL_PROB", "0.90"))
_ADAPTIVE_CTR_FLOOR_MIN = float(os.getenv("TEST_ADAPTIVE_CTR_FLOOR_MIN", "0.008"))
_ADAPTIVE_CTR_FLOOR_SCALE = float(os.getenv("TEST_ADAPTIVE_CTR_FLOOR_SCALE", "0.70"))
_FATIGUE_DROP_PCT = float(os.getenv("TEST_FATIGUE_DROP_PCT", "0.35"))
_EWMA_ALPHA = float(os.getenv("TEST_EWMA_ALPHA", "0.30"))

_ENV_MIN_SPEND_BEFORE_KILL = float(os.getenv("TEST_MIN_SPEND_BEFORE_KILL", "20"))  # EUR
_BANDIT_SAMPLE_COUNT = int(os.getenv("TEST_BANDIT_SAMPLE_COUNT", "32"))
_MAX_LAUNCHES_PER_TICK = int(os.getenv("TEST_MAX_LAUNCHES_PER_TICK", "8"))

# Inline tripwire to mirror rules.yaml: spend_no_purchase ≥ 32 EUR
_SPEND_NO_PURCHASE_EUR = float(os.getenv("TEST_SPEND_NO_PURCHASE_EUR", "32"))


# ------------------------- small helpers -------------------------

def _now_minute_key(prefix: str) -> str:
    # use account-local timezone for tick idempotency
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
    """
    Count purchase/add_to_cart *events* from 'actions' list.
    In Meta insights, actions[].value is the count; revenue lives in action_values.
    """
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


def _stable_pass(store: Any, entity_id: str, rule_key: str, condition: bool) -> bool:
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
        return val >= _CONSEC_TICKS_REQUIRED
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


def _update_ewma_ctr(store: Any, ad_id: str, ctr_today: float) -> float:
    k = f"{ad_id}::ewma_ctr"
    prev = _get_counter_float(store, k) or ctr_today
    ewma = _EWMA_ALPHA * ctr_today + (1.0 - _EWMA_ALPHA) * prev
    _set_counter_float(store, k, ewma)
    return ewma


def _fatigue_detect(store: Any, ad_id: str, ctr_today: float) -> bool:
    ewma = _update_ewma_ctr(store, ad_id, ctr_today)
    return ewma > 0 and ctr_today < (1.0 - _FATIGUE_DROP_PCT) * ewma


def _bayes_kill_prob(clicks: float, imps: float, floor: float) -> float:
    a = _BETA_PRIOR_A + max(0.0, clicks)
    b = _BETA_PRIOR_B + max(0.0, imps - clicks)
    below = 0
    n = _BANDIT_SAMPLE_COUNT
    for _ in range(n):
        sample = random.betavariate(a, b)
        if sample < floor:
            below += 1
    return below / max(1, n)


def _adaptive_ctr_floor(rows: List[Dict[str, Any]], min_impressions: int) -> float:
    ctrs = sorted((_ctr(r) for r in rows if _safe_f(r.get("impressions")) >= min_impressions), reverse=True)
    if not ctrs:
        return _ADAPTIVE_CTR_FLOOR_MIN
    mid = ctrs[len(ctrs) // 2]
    return max(_ADAPTIVE_CTR_FLOOR_MIN, mid * _ADAPTIVE_CTR_FLOOR_SCALE)


def _data_quality_sentry(row: Dict[str, Any]) -> Optional[str]:
    spend = _safe_f(row.get("spend"))
    purch, atc = _purchase_and_atc_counts(row)
    if spend >= 20 and purch == 0 and atc == 0:
        return f"Spend present but no actions — check tracking"
    return None


def _bandit_score_for_queue_item(store: Any, label: str) -> float:
    try:
        clicks = store.get_counter(f"qctr::{label}::clicks")
        imps = store.get_counter(f"qctr::{label}::imps")
    except Exception:
        clicks, imps = 0, 0
    a = _BETA_PRIOR_A + clicks
    b = _BETA_PRIOR_B + max(0, imps - clicks)
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


# ---------- pause/resume helpers & one-shot alerts ----------

def _pause_ad(meta: Any, ad_id: str) -> Any:
    # Preferred: client's own pause method
    if hasattr(meta, "pause_ad") and callable(getattr(meta, "pause_ad")):
        return meta.pause_ad(ad_id)
    # Direct Graph POST to the object path /{ad_id}
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
        # SDK fallback if available
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


# ---------------------------------------------------------------------------


def run_testing_tick(
    meta: Any,
    settings: Dict[str, Any],
    engine: Any,
    store: Any,
    queue_df: pd.DataFrame,
    set_supabase_status: Callable[[List[str], str], None],  # passed from main.py
) -> Dict[str, Any]:
    summary = {"kills": 0, "promotions": 0, "launched": 0, "fatigue_flags": 0, "data_quality_alerts": 0}
    try:
        tk = _now_minute_key("testing")
        if hasattr(store, "tick_seen") and store.tick_seen(tk):
            notify(f"ℹ️ [TEST] Tick {tk} already processed; skipping.")
            return summary
    except Exception:
        pass

    # --- Resolve minimums from YAML (with env fallbacks) ---
    tmin = (settings.get("testing") or {}).get("minimums") or {}
    min_impressions = int(tmin.get("min_impressions", _ENV_MIN_IMPRESSIONS))
    min_clicks = int(tmin.get("min_clicks", _ENV_MIN_CLICKS))  # allow 0 from YAML
    min_spend = float(tmin.get("min_spend_eur", tmin.get("min_spend", _ENV_MIN_SPEND)))

    tfair = (settings.get("testing") or {}).get("fairness") or {}
    min_spend_before_kill = float(tfair.get("min_spend_before_kill_eur", _ENV_MIN_SPEND_BEFORE_KILL))

    adset_id = settings["ids"]["testing_adset_id"]
    keep_live = int(settings["testing"]["keep_ads_live"])
    validation_campaign_id = settings["ids"]["validation_campaign_id"]

    copy_bank = _load_copy_bank(settings)
    copy_strategy = (settings.get("copy_bank") or {}).get("strategy", "round_robin")

    # -------- Build ACTIVE/PAUSED map (for re-arming pause alerts) --------
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

    # Any ad that is ACTIVE should reset the paused-alert flag (so a future pause alerts once)
    for aid, st in status_by_ad_id.items():
        if st == "ACTIVE":
            _set_paused_alerted(store, aid, 0)

    # -------- Fetch insights via MetaClient (today + lifetime) --------
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
            action_attribution_windows=list(_ATTR_WINDOWS),
            paginate=True,
            date_preset="today",
        )

        rows_lifetime = meta.get_ad_insights(
            level="ad",
            filtering=[{"field": "adset.id", "operator": "IN", "value": [adset_id]}],
            fields=["ad_id", "ad_name", "spend", "actions", "action_values"],
            action_attribution_windows=list(_ATTR_WINDOWS),
            date_preset="lifetime",
            paginate=True,
        )
    except Exception as e:
        notify(f"❗ [TEST] Failed to fetch insights: {e}")
        return summary

    lifetime_by_id: Dict[str, Dict[str, Any]] = {}
    for lr in rows_lifetime or []:
        lifetime_by_id[str(lr.get("ad_id") or "")] = lr

    ctr_floor = _adaptive_ctr_floor(rows_today, min_impressions)

    for r in rows_today:
        ad_id = r.get("ad_id")
        name = r.get("ad_name", "")
        if not ad_id:
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

        # Gate on minimums only if BOTH today's and lifetime spend are below the kill budget.
        if (spend_today < min_spend_before_kill and spend_life < min_spend_before_kill) and \
           not _meets_minimums(r, min_impressions, min_clicks, min_spend):
            continue

        dq = _data_quality_sentry(r)
        if dq:
            notify(f"🩺 [TEST] {name}: {dq}")
            summary["data_quality_alerts"] += 1

        if _fatigue_detect(store, ad_id, ctr_today):
            notify(f"🟡 [TEST] Fatigue signal on {name} (CTR drop vs trend).")
            summary["fatigue_flags"] += 1

        bayes_p = _bayes_kill_prob(_safe_f(r.get("clicks")), _safe_f(r.get("impressions")), ctr_floor)
        bayes_kill = bayes_p >= _BAYES_KILL_PROB
        try:
            kill, reason_engine = engine.should_kill_testing(r)
        except Exception:
            kill, reason_engine = False, ""

        # ---------------- Tripwire kill (once-per-pause alert) ----------------
        if spend_life >= _SPEND_NO_PURCHASE_EUR and purch_life == 0:
            reason = f"Lifetime spend≥€{int(_SPEND_NO_PURCHASE_EUR)} & no purchase"
            try:
                _pause_ad(meta, ad_id)
                # One-shot paused alert per event
                if _get_paused_alerted(store, ad_id) == 0:
                    try:
                        alert_kill("TEST", name, reason, {"CTR": f"{ctr_today:.2%}", "ROAS": f"{_roas(r):.2f}"})
                    except Exception:
                        notify(f"🛑 [TEST] Killed {name} — {reason}")
                    _set_paused_alerted(store, ad_id, 1)

                # log after notify
                store.log(
                    entity_type="ad",
                    entity_id=ad_id,
                    action="PAUSE",
                    reason=f"[TEST] {reason}",
                    level="warn",
                    stage="TEST",
                    rule_type="testing_kill_tripwire",
                    thresholds={"spend_no_purchase_eur": _SPEND_NO_PURCHASE_EUR},
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
            continue  # don't consider promotion after a kill

        # ---------------- Other kill paths (with stability) ----------------
        should_kill_other = kill or bayes_kill
        if should_kill_other and _stable_pass(store, ad_id, "kill_test", True):
            try:
                _pause_ad(meta, ad_id)
                reason = (
                    reason_engine
                    or ("CTR<{:.2%} (p={:.2f})".format(ctr_floor, bayes_p) if bayes_kill else "Rule-based kill")
                )

                # One-shot paused alert per event
                if _get_paused_alerted(store, ad_id) == 0:
                    try:
                        alert_kill("TEST", name, reason, {"CTR": f"{ctr_today:.2%}", "ROAS": f"{_roas(r):.2f}"})
                    except Exception:
                        notify(f"🛑 [TEST] Killed {name} — {reason}")
                    _set_paused_alerted(store, ad_id, 1)

                store.log(
                    entity_type="ad",
                    entity_id=ad_id,
                    action="PAUSE",
                    reason=f"[TEST] {reason}",
                    level="warn",
                    stage="TEST",
                    rule_type="testing_kill",
                    thresholds={"ctr_floor": ctr_floor, "bayes_prob": _BAYES_KILL_PROB},
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
            _stable_pass(store, ad_id, "kill_test", False)

        # ---------------- Promotion ----------------
        try:
            adv, adv_reason = engine.should_advance_from_testing(r)
        except Exception:
            adv, adv_reason = False, ""

        if adv and _stable_pass(store, ad_id, "adv_test", True):
            label = name.replace("[TEST]", "").strip() or f"Ad_{ad_id}"
            try:
                valid_as = meta.create_validation_adset(validation_campaign_id, label, daily_budget=40.0)
            except Exception:
                valid_as = None
            store.log(
                entity_type="ad",
                entity_id=ad_id,
                action="TEST_TO_VALID",
                reason=adv_reason,
                level="info",
                stage="VALID",
                meta={"validation_adset": valid_as},
            )
            try:
                alert_promote("TEST", "VALID", label, budget=40.0)
            except Exception:
                notify(f"✅ [TEST→VALID] {label} — {adv_reason} (Budget: {ACCOUNT_CURRENCY_SYMBOL}40/day)")
            if _PAUSE_AFTER_PROMOTION:
                try:
                    _pause_ad(meta, ad_id)
                    _set_paused_alerted(store, ad_id, 1)  # avoid duplicate pause alert for the auto-pause after promotion
                except Exception:
                    pass
            summary["promotions"] += 1
        else:
            _stable_pass(store, ad_id, "adv_test", False)

    # -------- Top-up launches --------
    # If we didn't fetch earlier (on error), try again so we can determine open slots.
    if not current_ads:
        try:
            current_ads = meta.list_ads_in_adset(adset_id)
        except Exception as e:
            notify(f"❗ [TEST] Could not list ads in ad set: {e}")
            return summary

    active_count = _active_count(current_ads)
    need = max(0, min(_MAX_LAUNCHES_PER_TICK, keep_live - active_count))
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
        key=lambda r: _bandit_score_for_queue_item(store, str(r.get("_label_core") or _label_from_row(r)).strip()),
        reverse=True,
    )
    picks = ranked[:need]

    for p in picks:
        label_core = str(p.get("_label_core") or _label_from_row(p)).strip()
        cname = f"[TEST] {label_core}"

        # Normalize & validate video_id
        video_id_raw = p.get("video_id")
        video_id = _normalize_video_id_cell(video_id_raw)
        if not label_core or not video_id or not video_id.isdigit():
            notify(f"⚠️ [TEST] Skipping '{label_core or 'UNNAMED'}' — missing/invalid video_id (got: {video_id_raw!r}).")
            continue

        try:
            primary_text, headline, description = _choose_mix_and_match(store, _load_copy_bank(settings), label_core, copy_strategy)
        except Exception as e:
            notify(f"⚠️ [TEST] Copy bank issue for '{label_core}': {e}")
            continue

        try:
            cc = engine.creative_compliance(
                {
                    "primary_text": primary_text,
                    "headline": headline,
                    "description": description,
                    "link_url": os.getenv("STORE_URL", ""),
                }
            )
            if not cc.get("ok", True):
                notify(f"🚫 [TEST] '{label_core}' failed compliance: {', '.join(cc.get('issues', []))}")
                continue
        except Exception:
            pass

        try:
            creative = meta.create_video_creative(
                page_id=p.get("page_id"),
                name=cname,
                video_library_id=video_id,
                primary_text=primary_text,
                headline=headline,
                description=description,
                call_to_action="SHOP_NOW",
                link_url=os.getenv("STORE_URL"),
                utm_params=p.get("utm_params") or None,
                thumbnail_url=p.get("thumbnail_url") or None,
            )
            ad = meta.create_ad(adset_id, cname, creative_id=creative["id"], status="ACTIVE")
            store.log(
                entity_type="ad",
                entity_id=ad["id"],
                action="CREATE",
                reason="Top-up to keep_ads_live",
                level="info",
                stage="TEST",
                meta={"testing_adset_id": adset_id},
            )
            notify(f"🟢 [TEST] Launched {label_core}")
            summary["launched"] += 1
            _record_queue_feedback(store, label_core, clicks=1, imps=200)

            # --- mark Supabase row as launched using your status column ---
            try:
                cid = str(p.get("creative_id") or "").strip()
                if cid:
                    set_supabase_status([cid], "launched")  # use PK column
                else:
                    vid = _normalize_video_id_cell(p.get("video_id"))
                    if vid:
                        set_supabase_status([vid], "launched")  # falls back to video_id if your helper supports it
            except Exception:
                pass

        except Exception as e:
            notify(f"❗ [TEST] Failed to launch '{label_core}': {e}")

    return summary
