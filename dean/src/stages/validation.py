from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from slack import alert_kill, alert_promote, notify

UTC = timezone.utc

# ---- Validation stage thresholds (EUR, Amsterdam account) ----
_MIN_IMPRESSIONS = int(os.getenv("VALID_MIN_IMPRESSIONS", "600"))
_MIN_CLICKS = int(os.getenv("VALID_MIN_CLICKS", "30"))
_MIN_SPEND = float(os.getenv("VALID_MIN_SPEND", "40"))  # €40 per your plan
_CONSEC_TICKS_REQUIRED = int(os.getenv("VALID_CONSEC_TICKS_REQUIRED", "2"))
_ATTR_WINDOWS = tuple((os.getenv("VALID_ATTR_WINDOWS", "7d_click,1d_view") or "7d_click,1d_view").split(","))
_PAUSE_AFTER_PROMOTION = (os.getenv("VALID_PAUSE_AFTER_PROMOTION", "1").lower() in ("1", "true", "yes"))

# Ensure we don't kill before reasonable validation spend
_MIN_FAIR_SPEND_BEFORE_KILL = float(os.getenv("VALID_MIN_FAIR_SPEND_BEFORE_KILL", "40"))  # €40

# Promotion behavior / soft pass (EUR)
_PROMO_COOLDOWN_HOURS = int(os.getenv("VALID_PROMO_COOLDOWN_HOURS", "24"))
_SOFT_PASS_MAX_BUDGET = float(os.getenv("VALID_SOFT_PASS_MAX_BUDGET", "40"))   # €40/day soft lane
_SOFT_PASS_MAX_CPA = float(os.getenv("VALID_SOFT_PASS_MAX_CPA", "36"))         # allow borderline CPA for soft lane
_SOFT_PASS_MIN_ROAS = float(os.getenv("VALID_SOFT_PASS_MIN_ROAS", "1.5"))
_SOFT_PASS_MIN_CTR = float(os.getenv("VALID_SOFT_PASS_MIN_CTR", "0.008"))
_ALLOW_CREATE_ON_PROMO_CHECK = (os.getenv("VALID_ALLOW_CREATE_ON_PROMO_CHECK", "0").lower() in ("1", "true", "yes"))

def _now_minute_key(prefix: str) -> str:
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

def _meets_minimums(row: Dict[str, Any]) -> bool:
    spend = _safe_f(row.get("spend"))
    imps = _safe_f(row.get("impressions"))
    clicks = _safe_f(row.get("clicks"))
    return spend >= _MIN_SPEND and imps >= _MIN_IMPRESSIONS and clicks >= _MIN_CLICKS

def _stable_pass(store: Any, entity_id: str, rule_key: str, condition: bool) -> bool:
    key = f"{entity_id}::stable::{rule_key}"
    try:
        if condition:
            return store.incr(key, 1) >= _CONSEC_TICKS_REQUIRED
        store.set_counter(key, 0)
        return False
    except Exception:
        return condition

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

def _purchase_days(meta: Any, ad_id: str, days: int = 7) -> int:
    try:
        rows = meta.get_ad_insights(
            level="ad",
            time_range=None,
            filtering=[{"field": "ad.id", "operator": "IN", "value": [ad_id]}],
            fields=["actions", "spend", "impressions", "clicks", "date_start"],
            action_attribution_windows=list(_ATTR_WINDOWS),
            breakdowns=["date"],
            paginate=True,
        )
        seen = set()
        for r in rows:
            acts = r.get("actions") or []
            purch = 0.0
            for a in acts:
                if a.get("action_type") == "purchase":
                    purch += _safe_f(a.get("value"))
            if purch > 0:
                d = (r.get("date_start") or r.get("date") or "").split("T")[0]
                if d:
                    seen.add(d)
        return min(len(seen), days)
    except Exception:
        return 0

def _adaptive_start_budget_eur(row: Dict[str, Any], default_eur: float) -> float:
    """Pick a sensible starting budget in EUR based on observed CPA (caps included)."""
    spend = _safe_f(row.get("spend"))
    purchases = 0.0
    for a in (row.get("actions") or []):
        if a.get("action_type") == "purchase":
            purchases += _safe_f(a.get("value"))
    cpa = (spend / purchases) if purchases > 0 else None
    if cpa is None:
        return float(default_eur)
    return float(max(_SOFT_PASS_MAX_BUDGET, min(3.0 * cpa, 150.0)))

def _promotion_cooldown_active(store: Any, ad_id: str) -> bool:
    try:
        rec = store.get_flag("ad", ad_id, "promo_cooldown_until")
        until_iso = rec.get("v") if isinstance(rec, dict) else rec
        if until_iso:
            try:
                return datetime.fromisoformat(str(until_iso)) > datetime.now(UTC)
            except Exception:
                return False
    except Exception:
        return False
    return False

def _set_promotion_cooldown(store: Any, ad_id: str) -> None:
    try:
        until = (datetime.now(UTC) + timedelta(hours=_PROMO_COOLDOWN_HOURS)).isoformat()
        store.set_flag("ad", ad_id, "promo_cooldown_until", until)
    except Exception:
        pass

def _eligible_soft_pass(row: Dict[str, Any]) -> bool:
    spend = _safe_f(row.get("spend"))
    ctr = _safe_ctr(row)
    roas = _safe_roas(row)
    purchases = 0.0
    for a in (row.get("actions") or []):
        if a.get("action_type") == "purchase":
            purchases += _safe_f(a.get("value"))
    cpa = (spend / purchases) if purchases > 0 else None
    return purchases >= 1 and ((cpa is not None and cpa <= _SOFT_PASS_MAX_CPA) or ctr >= _SOFT_PASS_MIN_CTR or roas >= _SOFT_PASS_MIN_ROAS)

def _find_existing_scaling_adset(meta: Any, scaling_campaign_id: str, creative_label: str) -> Optional[str]:
    if not hasattr(meta, "ensure_adset") or not isinstance(creative_label, str):
        return None
    if not _ALLOW_CREATE_ON_PROMO_CHECK:
        return None
    try:
        res = meta.ensure_adset(
            campaign_id=scaling_campaign_id,
            name=f"[SCALE] {creative_label}",
            daily_budget_usd=0.0,  # value ignored; we only probe existence
            status="PAUSED",
        )
        return res.get("id")
    except Exception:
        return None

def run_validation_tick(meta: Any, settings: Dict[str, Any], engine: Any, store: Any) -> Dict[str, int]:
    summary = {"kills": 0, "promotions": 0, "soft_passes": 0}
    try:
        if hasattr(store, "tick_seen") and store.tick_seen(_now_minute_key("validation")):
            notify("ℹ️ [VALID] Skipping duplicate tick.")
            return summary
    except Exception:
        pass

    validation_campaign_id = settings["ids"]["validation_campaign_id"]
    scaling_campaign_id = settings["ids"]["scaling_campaign_id"]

    # Start budget interpreted in EUR (ad account currency). Kept as *_usd in Meta client API for compatibility.
    start_budget_default_eur = float(settings.get("scaling", {}).get("adset_start_budget_eur", 100.0))

    try:
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
            action_attribution_windows=list(_ATTR_WINDOWS),
            paginate=True,
        )
    except Exception as e:
        notify(f"❗ [VALID] Insights error: {e}")
        return summary

    for r in rows:
        ad_id = r.get("ad_id")
        ad_name = r.get("ad_name", "")
        if not ad_id:
            continue

        # Ensure minimum fair spend before a kill
        if _safe_f(r.get("spend")) < _MIN_FAIR_SPEND_BEFORE_KILL:
            pass  # allow promotion checks below, but defer killing

        if not _meets_minimums(r):
            continue

        pdays = _purchase_days(meta, ad_id)
        if pdays:
            r["purchase_days"] = pdays

        # KILL checks
        try:
            kill, reason = engine.should_kill_validation(r)
        except Exception:
            kill, reason = False, ""
        if kill and _stable_pass(store, ad_id, "kill_valid", True):
            if _safe_f(r.get("spend")) < _MIN_FAIR_SPEND_BEFORE_KILL:
                _stable_pass(store, ad_id, "kill_valid", False)
            else:
                try:
                    meta.pause_ad(ad_id)
                    store.log_kill(
                        stage="VALID",
                        entity_id=ad_id,
                        rule_type="validation_kill",
                        reason=reason,
                        observed={
                            "spend": _safe_f(r.get("spend")),
                            "CTR": round(_safe_ctr(r), 4),
                            "ROAS": round(_safe_roas(r), 3),
                            "purchase_days": r.get("purchase_days", 0),
                        },
                        thresholds={"min_spend": _MIN_SPEND, "min_imps": _MIN_IMPRESSIONS, "min_clicks": _MIN_CLICKS},
                    )
                    try:
                        alert_kill(
                            "VALID",
                            ad_name,
                            reason,
                            {"spend": f"{_safe_f(r.get('spend')):.0f}", "CTR": f"{_safe_ctr(r):.2%}", "ROAS": f"{_safe_roas(r):.2f}"},
                        )
                    except Exception:
                        notify(f"🛑 [VALID] Killed {ad_name} — {reason}")
                    summary["kills"] += 1
                except Exception as e:
                    notify(f"❗ [VALID] Pause failed for {ad_name}: {e}")
            continue
        else:
            _stable_pass(store, ad_id, "kill_valid", False)

        # PROMOTE checks
        if _promotion_cooldown_active(store, ad_id):
            continue
        try:
            adv, adv_reason = engine.should_advance_from_validation(r)
        except Exception:
            adv, adv_reason = False, ""
        if adv and _stable_pass(store, ad_id, "adv_valid", True):
            creative_label = ad_name.replace("[VALID]", "").strip() or f"Ad_{ad_id}"

            # Avoid creating duplicate scaling ad sets
            exists = _find_existing_scaling_adset(meta, scaling_campaign_id, creative_label)
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

            start_budget = _adaptive_start_budget_eur(r, start_budget_default_eur)
            try:
                # NOTE: Meta client still expects parameter name *_usd; value is in EUR because the ad account currency is EUR.
                if hasattr(meta, "create_scaling_adset"):
                    scaling_adset = meta.create_scaling_adset(scaling_campaign_id, creative_label, daily_budget_usd=start_budget)
                elif hasattr(meta, "ensure_adset"):
                    scaling_adset = meta.ensure_adset(campaign_id=scaling_campaign_id, name=f"[SCALE] {creative_label}", daily_budget_usd=start_budget)
                else:
                    scaling_adset = {"id": None, "name": f"[SCALE] {creative_label}"}

                created_ad = None
                creative_id = None
                try:
                    if hasattr(meta, "get_ad_creative_id"):
                        creative_id = meta.get_ad_creative_id(ad_id)
                    if creative_id and hasattr(meta, "create_ad") and scaling_adset.get("id"):
                        created_ad = meta.create_ad(scaling_adset["id"], f"[SCALE] {creative_label}", creative_id=creative_id, status="ACTIVE")
                except Exception:
                    created_ad = None

                try:
                    store.log_promote(
                        from_stage="VALID",
                        to_stage="SCALE",
                        entity_id=ad_id,
                        reason=adv_reason,
                        meta={"scaling_adset": scaling_adset, "scaling_ad": created_ad, "start_budget_eur": start_budget},
                    )
                except Exception:
                    pass
                try:
                    # alert_promote uses a $ symbol in its template; we still send the numeric value.
                    alert_promote("VALID", "SCALE", creative_label, budget=start_budget)
                except Exception:
                    notify(f"🚀 [VALID→SCALE] {creative_label} — {adv_reason} (start €{start_budget:.0f}/d)")
                if _PAUSE_AFTER_PROMOTION:
                    try:
                        meta.pause_ad(ad_id)
                    except Exception:
                        pass
                _set_promotion_cooldown(store, ad_id)
                summary["promotions"] += 1
            except Exception as e:
                notify(f"❗ [VALID] Promotion failed for {ad_name}: {e}")
        else:
            _stable_pass(store, ad_id, "adv_valid", False)

        # SOFT PASS lane (keep small budget pressure on borderline winners)
        if summary["promotions"] == 0 and _eligible_soft_pass(r):
            soft_budget = min(_SOFT_PASS_MAX_BUDGET, start_budget_default_eur)
            try:
                label = ad_name.replace("[VALID]", "").strip() or f"Ad_{ad_id}"
                if hasattr(meta, "create_scaling_adset"):
                    aset = meta.create_scaling_adset(scaling_campaign_id, f"{label} • Soft", daily_budget_usd=soft_budget)
                elif hasattr(meta, "ensure_adset"):
                    aset = meta.ensure_adset(campaign_id=scaling_campaign_id, name=f"[SCALE] {label} • Soft", daily_budget_usd=soft_budget)
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
                summary["soft_passes"] += 1
            except Exception:
                pass

    return summary
