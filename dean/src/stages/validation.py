from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from slack import alert_kill, alert_promote, alert_error, notify

UTC = timezone.utc

# ---- Local timezone (Amsterdam) for day buckets ----
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

ACCOUNT_TZ_NAME = os.getenv("ACCOUNT_TZ") or os.getenv("ACCOUNT_TIMEZONE") or "Europe/Amsterdam"
LOCAL_TZ = ZoneInfo(ACCOUNT_TZ_NAME) if ZoneInfo else None
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


def _getenv_b(name: str, default: bool) -> bool:
    try:
        raw = os.getenv(name, str(int(default))).lower()
        return raw in ("1", "true", "yes", "y")
    except Exception:
        return default


def _cfg(settings: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = settings
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _cfg_or_env_i(settings: Dict[str, Any], path: str, env: str, default: int) -> int:
    v = _cfg(settings, path, None)
    if v is None:
        return _getenv_i(env, default)
    try:
        return int(v)
    except Exception:
        return default


def _cfg_or_env_f(settings: Dict[str, Any], path: str, env: str, default: float) -> float:
    v = _cfg(settings, path, None)
    if v is None:
        return _getenv_f(env, default)
    try:
        return float(v)
    except Exception:
        return default


def _cfg_or_env_b(settings: Dict[str, Any], path: str, env: str, default: bool) -> bool:
    v = _cfg(settings, path, None)
    if v is None:
        return _getenv_b(env, default)
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

def _now_local() -> datetime:
    return datetime.now(LOCAL_TZ) if LOCAL_TZ else datetime.now(UTC)


def _today_str() -> str:
    return _now_local().strftime("%Y-%m-%d")


def _daily_key(stage: str, metric: str) -> str:
    # daily::<YYYY-MM-DD>::STAGE::metric
    return f"daily::{_today_str()}::{stage}::{metric}"


def _now_minute_key(prefix: str) -> str:
    # Validation ticks remain UTC-based for idempotency; day totals use local time.
    return f"{prefix}::{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M')}"


def _safe_f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v).replace(",", "").strip())
    except Exception:
        return default


def _safe_ctr(row: Dict[str, Any]) -> float:
    imps = _safe_f(row.get("impressions"))
    clicks = _safe_f(row.get("clicks"))
    return (clicks / imps) if imps > 0 else 0.0


def _safe_roas(row: Dict[str, Any]) -> float:
    roas_list = row.get("purchase_roas") or []
    try:
        return float(roas_list[0].get("value", 0)) if roas_list else 0.0
    except Exception:
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


def _meets_minimums(row: Dict[str, Any], min_spend: float, min_imps: int, min_clicks: int) -> bool:
    spend = _safe_f(row.get("spend"))
    imps = _safe_f(row.get("impressions"))
    clicks = _safe_f(row.get("clicks"))
    return spend >= min_spend and imps >= min_imps and clicks >= min_clicks


def _stable_pass(store: Any, entity_id: str, rule_key: str, condition: bool, consec_required: int) -> bool:
    key = f"{entity_id}::stable::{rule_key}"
    try:
        if condition:
            return store.incr(key, 1) >= consec_required
        store.set_counter(key, 0)
        return False
    except Exception:
        return condition


def _purchase_days(meta: Any, ad_id: str, attr_windows: List[str], days: int = 7) -> int:
    """
    Count distinct days with >= 1 purchase over the last N days using date breakdown.
    """
    try:
        since = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
        until = datetime.now(UTC).strftime("%Y-%m-%d")
        rows = meta.get_ad_insights(
            level="ad",
            time_range={"since": since, "until": until},
            filtering=[{"field": "ad.id", "operator": "IN", "value": [ad_id]}],
            fields=["actions", "date_start"],
            action_attribution_windows=list(attr_windows),
            breakdowns=["date"],
            paginate=True,
        )
        seen = set()
        for r in rows:
            acts = r.get("actions") or []
            if any(a.get("action_type") == "purchase" and _safe_f(a.get("value")) > 0 for a in acts):
                d = (r.get("date_start") or "").split("T")[0]
                if d:
                    seen.add(d)
        return len(seen)
    except Exception:
        return 0


def _adaptive_start_budget_eur(row: Dict[str, Any], default_eur: float, soft_max_budget: float) -> float:
    """
    Choose a starting budget in EUR based on observed CPA with caps.
    If no purchases, fall back to default.
    Caps at soft_max_budget and at 150 EUR, and scales roughly as 3x CPA.
    """
    spend = _safe_f(row.get("spend"))
    purchases = 0.0
    for a in (row.get("actions") or []):
        if a.get("action_type") == "purchase":
            purchases += _safe_f(a.get("value"))
    if purchases <= 0:
        return float(default_eur)
    cpa = spend / purchases
    return float(min(soft_max_budget, 3.0 * cpa, 150.0))


def _promotion_cooldown_active(store: Any, ad_id: str) -> bool:
    try:
        rec = store.get_flag("ad", ad_id, "promo_cooldown_until")
        until_iso = rec.get("v") if isinstance(rec, dict) else rec
        if until_iso:
            return datetime.fromisoformat(str(until_iso)) > datetime.now(UTC)
    except Exception:
        pass
    return False


def _set_promotion_cooldown(store: Any, ad_id: str, hours: int) -> None:
    try:
        until = (datetime.now(UTC) + timedelta(hours=hours)).isoformat()
        store.set_flag("ad", ad_id, "promo_cooldown_until", until)
    except Exception:
        pass


# ---- Soft pass cooldown helpers ----

def _soft_pass_cooldown_active(store: Any, ad_id: str) -> bool:
    try:
        rec = store.get_flag("ad", ad_id, "soft_pass_cooldown_until")
        until_iso = rec.get("v") if isinstance(rec, dict) else rec
        if until_iso:
            return datetime.fromisoformat(str(until_iso)) > datetime.now(UTC)
    except Exception:
        pass
    return False


def _set_soft_pass_cooldown(store: Any, ad_id: str, hours: int) -> None:
    try:
        until = (datetime.now(UTC) + timedelta(hours=hours)).isoformat()
        store.set_flag("ad", ad_id, "soft_pass_cooldown_until", until)
    except Exception:
        pass


def _eligible_soft_pass(row: Dict[str, Any], max_cpa: float, min_ctr: float, min_roas: float) -> bool:
    spend = _safe_f(row.get("spend"))
    ctr = _safe_ctr(row)
    roas = _safe_roas(row)
    purchases = 0.0
    for a in (row.get("actions") or []):
        if a.get("action_type") == "purchase":
            purchases += _safe_f(a.get("value"))
    cpa = (spend / purchases) if purchases > 0 else None
    return purchases >= 1 and (
        (cpa is not None and cpa <= max_cpa) or ctr >= min_ctr or roas >= min_roas
    )


def _find_existing_scaling_adset(meta: Any, scaling_campaign_id: str, creative_label: str, allow_probe: bool) -> Optional[str]:
    """
    Best-effort probe to avoid duplicates.
    If allow_probe is False, do nothing.
    If your client ensure_adset creates when missing, keep allow_probe False to avoid side effects.
    """
    if not hasattr(meta, "ensure_adset") or not isinstance(creative_label, str):
        return None
    if not allow_probe:
        return None
    try:
        res = meta.ensure_adset(
            campaign_id=scaling_campaign_id,
            name=f"[SCALE] {creative_label}",
            daily_budget=0.0,  # value ignored by probe
            status="PAUSED",
        )
        return res.get("id")
    except Exception:
        return None


# ---------------------------------------------------------------------------

def run_validation_tick(meta: Any, settings: Dict[str, Any], engine: Any, store: Any) -> Dict[str, int]:
    summary = {"kills": 0, "promotions": 0, "soft_passes": 0}
    try:
        if hasattr(store, "tick_seen") and store.tick_seen(_now_minute_key("validation")):
            notify("ℹ️ [VALID] Skipping duplicate tick.")
            return summary
    except Exception:
        pass

    # IDs
    validation_campaign_id = _cfg(settings, "ids.validation_campaign_id")
    scaling_campaign_id = _cfg(settings, "ids.scaling_campaign_id")

    # ----- Config (YAML with env fallbacks) -----
    # Minimums
    min_imps = _cfg_or_env_i(settings, "validation.minimums.min_impressions", "VALID_MIN_IMPRESSIONS", 600)
    min_clicks = _cfg_or_env_i(settings, "validation.minimums.min_clicks", "VALID_MIN_CLICKS", 30)
    min_spend = _cfg_or_env_f(settings, "validation.minimums.min_spend", "VALID_MIN_SPEND", 40.0)

    # Engine
    attr_windows = _cfg_or_env_list(settings, "validation.engine.attribution_windows", "VALID_ATTR_WINDOWS", ["7d_click", "1d_view"])
    consec_required = _cfg_or_env_i(
        settings,
        "validation.engine.stability.consecutive_ticks",
        "VALID_CONSEC_TICKS_REQUIRED",
        _cfg_or_env_i(settings, "stability.consecutive_ticks", "VALID_CONSEC_TICKS_REQUIRED", 2),
    )
    fair_min_spend_before_kill = _cfg_or_env_f(settings, "validation.engine.fairness.min_spend_before_kill_eur", "VALID_MIN_FAIR_SPEND_BEFORE_KILL", 40.0)
    tripwire_life_spend_no_purchase = _cfg_or_env_f(settings, "validation.engine.tripwires.lifetime_spend_no_purchase_eur", "VALID_SPEND_NO_PURCHASE_EUR", fair_min_spend_before_kill)

    pause_after_promotion = _cfg_or_env_b(settings, "validation.engine.promotion.pause_after_promotion", "VALID_PAUSE_AFTER_PROMOTION", True)
    promo_cooldown_hours = _cfg_or_env_i(settings, "validation.engine.promotion.cooldown_hours", "VALID_PROMO_COOLDOWN_HOURS", 24)
    allow_create_on_promo_check = _cfg_or_env_b(settings, "validation.engine.promotion.allow_create_on_promo_check", "VALID_ALLOW_CREATE_ON_PROMO_CHECK", False)

    soft_max_budget = _cfg_or_env_f(settings, "validation.engine.soft_pass.max_budget_eur", "VALID_SOFT_PASS_MAX_BUDGET", 40.0)
    soft_max_cpa = _cfg_or_env_f(settings, "validation.engine.soft_pass.max_cpa_eur", "VALID_SOFT_PASS_MAX_CPA", 36.0)
    soft_min_roas = _cfg_or_env_f(settings, "validation.engine.soft_pass.min_roas", "VALID_SOFT_PASS_MIN_ROAS", 1.5)
    soft_min_ctr = _cfg_or_env_f(settings, "validation.engine.soft_pass.min_ctr", "VALID_SOFT_PASS_MIN_CTR", 0.008)
    soft_pass_cooldown_hours = _cfg_or_env_i(settings, "validation.engine.soft_pass.cooldown_hours", "VALID_SOFT_PASS_COOLDOWN_HOURS", 24)

    # Scaling start budget (used on promotion)
    start_budget_default_eur = _cfg_or_env_f(settings, "scaling.adset_start_budget_eur", "VALID_START_BUDGET_EUR", 100.0)

    # ----- Fetch TODAY (default window) and LIFETIME (since ad launch) -----
    try:
        # Today/default — used for performance metrics, promotion logic, etc.
        rows = meta.get_ad_insights(
            level="ad",
            filtering=[{"field": "campaign.id", "operator": "IN", "value": [validation_campaign_id]}],
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
        )
        # Lifetime — used to decide kills on total spend since launch
        rows_life = meta.get_ad_insights(
            level="ad",
            filtering=[{"field": "campaign.id", "operator": "IN", "value": [validation_campaign_id]}],
            fields=["ad_id", "spend", "actions", "action_values"],
            action_attribution_windows=list(attr_windows),
            date_preset="lifetime",
            paginate=True,
        )
    except Exception as e:
        notify(f"❗ [VALID] Insights error: {e}")
        return summary

    # Map lifetime rows by ad_id
    life_by_ad: Dict[str, Dict[str, Any]] = {}
    for lr in rows_life or []:
        aid = str(lr.get("ad_id") or "")
        if aid:
            life_by_ad[aid] = lr

    for r in rows:
        ad_id = r.get("ad_id")
        ad_name = r.get("ad_name", "")
        if not ad_id:
            continue

        # Today metrics
        spend_today = _safe_f(r.get("spend"))
        ctr_today = _safe_ctr(r)
        roas_today = _safe_roas(r)
        purch_today, _atc_today = _purchase_and_atc_counts(r)

        # Lifetime metrics
        lr = life_by_ad.get(str(ad_id), {}) or {}
        spend_life = _safe_f(lr.get("spend"))
        purch_life, _atc_life = _purchase_and_atc_counts(lr)

        # Removed per-ad informational messages - now handled in consolidated run summary

        # Fairness gating
        if (spend_today < fair_min_spend_before_kill and spend_life < fair_min_spend_before_kill) and not _meets_minimums(r, min_spend, min_imps, min_clicks):
            continue

        # Informational signal
        pdays = _purchase_days(meta, ad_id, attr_windows)
        if pdays:
            r["purchase_days"] = pdays

        # ---- KILL checks ----
        try:
            kill_engine, reason_engine = engine.should_kill_validation(r)
        except Exception:
            kill_engine, reason_engine = False, ""

        kill_tripwire = (spend_life >= tripwire_life_spend_no_purchase and purch_life == 0)
        kill_condition = bool(kill_engine or kill_tripwire)

        if kill_condition and _stable_pass(store, ad_id, "kill_valid", True, consec_required):
            # If still below fairness lifetime spend, reset and skip
            if spend_life < fair_min_spend_before_kill:
                _stable_pass(store, ad_id, "kill_valid", False, consec_required)
            else:
                reason = reason_engine or (f"Lifetime spend≥{ACCOUNT_CURRENCY_SYMBOL}{int(tripwire_life_spend_no_purchase)} & no purchase" if kill_tripwire else "Validation rule failed")
                try:
                    if hasattr(meta, "pause_ad"):
                        meta.pause_ad(ad_id)
                    else:
                        raise RuntimeError("meta.pause_ad not available")

                    # daily counter
                    try:
                        store.incr(_daily_key("VALID", "kills"), 1)
                    except Exception:
                        pass

                    # structured log
                    try:
                        store.log_kill(
                            stage="VALID",
                            entity_id=ad_id,
                            rule_type="validation_kill",
                            reason=reason,
                            observed={
                                "spend_today": spend_today,
                                "spend_lifetime": spend_life,
                                "purchases_today": purch_today,
                                "purchases_lifetime": purch_life,
                                "CTR": round(ctr_today, 4),
                                "ROAS": round(roas_today, 3),
                                "purchase_days": r.get("purchase_days", 0),
                            },
                            thresholds={
                                "min_spend": min_spend,
                                "min_imps": min_imps,
                                "min_clicks": min_clicks,
                                "lifetime_tripwire": tripwire_life_spend_no_purchase,
                            },
                        )
                    except Exception:
                        pass

                    # alert
                    try:
                        alert_kill(
                            "VALID",
                            ad_name,
                            reason,
                            {"spend_life": f"{spend_life:.0f}", "CTR": f"{ctr_today:.2%}", "ROAS": f"{roas_today:.2f}"},
                        )
                    except Exception:
                        alert_kill(
                            "VALID",
                            ad_name,
                            reason,
                            {"spend_life": f"{spend_life:.0f}", "CTR": f"{ctr_today:.2%}", "ROAS": f"{roas_today:.2f}"},
                        )
                    summary["kills"] += 1
                except Exception as e:
                    notify(f"❗ [VALID] Pause failed for {ad_name}: {e}")
            continue
        else:
            _stable_pass(store, ad_id, "kill_valid", False, consec_required)

        # ---- PROMOTE checks ----
        if _promotion_cooldown_active(store, ad_id):
            continue
        try:
            adv, adv_reason = engine.should_advance_from_validation(r)
        except Exception:
            adv, adv_reason = False, ""
        if adv and _stable_pass(store, ad_id, "adv_valid", True, consec_required):
            creative_label = ad_name.replace("[VALID]", "").strip() or f"Ad_{ad_id}"

            # Avoid duplicate scaling ad sets
            exists = _find_existing_scaling_adset(meta, scaling_campaign_id, creative_label, allow_probe=allow_create_on_promo_check)
            if exists:
                try:
                    store.log(
                        entity_type="ad",
                        entity_id=ad_id,
                        action="PROMOTE_SKIPPED_DUP",
                        reason=f"Scaling ad set exists ({creative_label})",
                        level="info",
                        stage="VALID",
                        meta={"scaling_adset_id": exists},
                    )
                except Exception:
                    pass
                continue

            start_budget = _adaptive_start_budget_eur(r, start_budget_default_eur, soft_max_budget)
            try:
                # Create scaling ad set
                if hasattr(meta, "create_scaling_adset"):
                    scaling_adset = meta.create_scaling_adset(scaling_campaign_id, creative_label, daily_budget=start_budget)
                elif hasattr(meta, "ensure_adset"):
                    scaling_adset = meta.ensure_adset(
                        campaign_id=scaling_campaign_id,
                        name=f"[SCALE] {creative_label}",
                        daily_budget=start_budget,
                    )
                else:
                    scaling_adset = {"id": None, "name": f"[SCALE] {creative_label}"}

                created_ad = None
                creative_id = None
                try:
                    if hasattr(meta, "get_ad_creative_id"):
                        creative_id = meta.get_ad_creative_id(ad_id)
                    if creative_id and scaling_adset.get("id"):
                        # Use promote_ad_with_continuity to maintain the same ad ID across stages
                        if hasattr(meta, "promote_ad_with_continuity"):
                            created_ad = meta.promote_ad_with_continuity(
                                original_ad_id=ad_id,
                                new_adset_id=scaling_adset["id"],
                                new_name=f"[SCALE] {creative_label}",
                                creative_id=creative_id,
                                status="ACTIVE",
                            )
                        elif hasattr(meta, "create_ad"):
                            # Fallback to regular create_ad if promote_ad_with_continuity is not available
                            created_ad = meta.create_ad(
                                scaling_adset["id"],
                                f"[SCALE] {creative_label}",
                                creative_id=creative_id,
                                status="ACTIVE",
                            )
                except Exception:
                    created_ad = None

                # daily counter
                try:
                    store.incr(_daily_key("VALID", "promotions"), 1)
                except Exception:
                    pass

                try:
                    # Include ID continuity information in the promotion log
                    promotion_meta = {
                        "scaling_adset": scaling_adset, 
                        "scaling_ad": created_ad, 
                        "start_budget_eur": start_budget,
                        "id_continuity": True,  # Flag indicating this ad maintains the same ID
                        "original_ad_id": ad_id,
                        "promoted_ad_id": created_ad.get("id") if created_ad else None,
                    }
                    store.log_promote(
                        from_stage="VALID",
                        to_stage="SCALE",
                        entity_id=ad_id,
                        reason=adv_reason,
                        meta=promotion_meta,
                    )
                except Exception:
                    pass
                try:
                    alert_promote("VALID", "SCALE", creative_label, budget=start_budget)
                except Exception:
                    alert_promote("VALID", "SCALE", creative_label, budget=start_budget)
                if pause_after_promotion:
                    try:
                        meta.pause_ad(ad_id)
                    except Exception:
                        pass
                _set_promotion_cooldown(store, ad_id, hours=promo_cooldown_hours)
                summary["promotions"] += 1
            except Exception as e:
                notify(f"❗ [VALID] Promotion failed for {ad_name}: {e}")
        else:
            _stable_pass(store, ad_id, "adv_valid", False, consec_required)

        # ---- SOFT PASS lane with cooldown (does not block promotions) ----
        if not _soft_pass_cooldown_active(store, ad_id) and _eligible_soft_pass(
            r, max_cpa=soft_max_cpa, min_ctr=soft_min_ctr, min_roas=soft_min_roas
        ):
            soft_budget = min(soft_max_budget, start_budget_default_eur)
            try:
                label = ad_name.replace("[VALID]", "").strip() or f"Ad_{ad_id}"
                if hasattr(meta, "create_scaling_adset"):
                    aset = meta.create_scaling_adset(scaling_campaign_id, f"{label} • Soft", daily_budget=soft_budget)
                elif hasattr(meta, "ensure_adset"):
                    aset = meta.ensure_adset(
                        campaign_id=scaling_campaign_id,
                        name=f"[SCALE] {label} • Soft",
                        daily_budget=soft_budget,
                    )
                else:
                    aset = {"id": None}
                try:
                    store.log(
                        entity_type="ad",
                        entity_id=ad_id,
                        action="SOFT_PASS",
                        reason="Borderline passed; created soft lane",
                        level="info",
                        stage="VALID",
                        meta={"adset": aset, "budget_eur": soft_budget},
                    )
                except Exception:
                    pass
                _set_soft_pass_cooldown(store, ad_id, hours=soft_pass_cooldown_hours)
                summary["soft_passes"] += 1
            except Exception:
                pass

    return summary
