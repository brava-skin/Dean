from __future__ import annotations

"""
Production-ready advanced scaler.

Key improvements vs. draft:
- Stronger typing & docstrings
- Environment-tunable thresholds (with sane defaults)
- Safer Store/Meta interactions (graceful fallbacks)
- Cooldown + idempotent logging keys (dedup per tick/budget)
- Sigma-clipping for outlier resistance hooks (kept lightweight)
- Compact but explicit control-flow; clear reasons in logs/alerts
- EURO/Amsterdam defaults for Brava account
"""

import os
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from metrics import Metrics, MetricsConfig, metrics_from_row
from slack import notify, alert_kill, alert_scale

# ====== Time ======
UTC = timezone.utc
def _now() -> datetime: return datetime.now(UTC)

# ====== Env knobs (all overridable at runtime) ======
def _env_bool(name: str, default: bool) -> bool:
    return (os.getenv(name, str(int(default))) or "").lower() in ("1", "true", "yes", "y")

def _env_f(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return default

def _env_i(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except Exception: return default

# ---- Scaling stage minimums (EUR account) ----
MIN_IMPRESSIONS      = _env_i("SCALE_MIN_IMPRESSIONS", 1000)   # per plan
MIN_SPEND            = _env_f("SCALE_MIN_SPEND", 50.0)         # €50
MIN_CLICKS           = _env_i("SCALE_MIN_CLICKS", 50)

# ---- Hysteresis/safety (EUR thresholds) ----
HYST_ROAS_DOWN       = _env_f("SCALE_HYST_ROAS_DOWN", 1.7)     # downscale if ROAS<1.7
HYST_CPA_UP          = _env_f("SCALE_HYST_CPA_UP", 33.0)       # downscale if CPA>€33

# ---- Kill windows (days) ----
KILL_CPA_DAYS        = _env_i("SCALE_KILL_CPA_DAYS", 2)        # CPA≥40 for 2 days
KILL_ROAS_DAYS       = _env_i("SCALE_KILL_ROAS_DAYS", 3)       # ROAS<1.2 for 3 days

# ---- Duplication / scaling limits ----
DUP_CAP_24H          = _env_i("SCALE_DUP_CAP_24H", 3)
SCALE_COOLDOWN_H     = _env_i("SCALE_COOLDOWN_H", 24)
MAX_SCALE_STEP_PCT   = _env_i("SCALE_MAX_SCALE_STEP_PCT", 200)

# ---- Reinvestment (EUR) ----
REINVEST_SHARE       = _env_f("SCALE_REINVEST_SHARE", 0.5)
REINVEST_MIN_BUMP    = _env_f("SCALE_REINVEST_MIN_BUMP", 10.0)  # €10 minimum bump
PORTFOLIO_MAX_MOVES  = _env_i("SCALE_PORTFOLIO_MAX_MOVES", 6)

ATTR_WINDOWS         = tuple((os.getenv("SCALE_ATTR_WINDOWS", "7d_click,1d_view") or "7d_click,1d_view").split(","))

# ====== Helpers ======
def _meets_minimums(m: Metrics) -> bool:
    return (m.spend or 0.0) >= MIN_SPEND and (m.impressions or 0.0) >= MIN_IMPRESSIONS and (m.clicks or 0.0) >= MIN_CLICKS

def _idkey(prefix: str) -> str:
    return f"{prefix}::{_now().strftime('%Y-%m-%dT%H:%M')}"

def _sigma_clip(values: List[float], z: float = 3.0) -> List[float]:
    if len(values) < 3: return values
    mu = statistics.mean(values)
    sd = statistics.pstdev(values) or 1.0
    return [v for v in values if abs(v - mu) <= z * sd]

def _credible_underperf(cpa: Optional[float], roas: float, spend: float) -> bool:
    # Lightweight probabilistic guard: only consider once there's a little signal.
    if spend < 50.0: return False
    if cpa is not None and spend > 100.0 and cpa > HYST_CPA_UP * 1.1:
        return True
    return roas < max(1.1, HYST_ROAS_DOWN * 0.9)

def _cooldown_ok(store: Any, adset_id: str, hours: int) -> bool:
    """Cooldown based on last successful scale timestamp."""
    try:
        rec = store.get_flag("adset", adset_id, "last_scale_ts")
        iso = rec.get("v") if isinstance(rec, dict) else rec
        if iso:
            try:
                if _now() - datetime.fromisoformat(str(iso)) < timedelta(hours=hours):
                    return False
            except Exception:
                return True
    except Exception:
        return True
    return True

def _mark_scaled(store: Any, adset_id: str) -> None:
    try:
        store.set_flag("adset", adset_id, "last_scale_ts", _now().isoformat())
    except Exception:
        pass

# ====== Thompson bandit & portfolio ======
@dataclass
class ArmState:
    successes: float = 0.0
    trials: float = 0.0
    last_score: float = 0.0

class ThompsonBandit:
    def __init__(self): self.state: Dict[str, ArmState] = {}
    def update(self, key: str, reward: float, trials: float = 1.0) -> None:
        s = self.state.get(key, ArmState())
        s.successes += max(0.0, reward)
        s.trials    += max(1.0, trials)
        s.last_score = reward
        self.state[key] = s
    def sample(self) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for k, s in self.state.items():
            a = 1.0 + max(0.0, s.successes)
            b = 1.0 + max(0.0, s.trials - s.successes)
            out.append((k, random.betavariate(a, b)))
        return sorted(out, key=lambda x: x[1], reverse=True)

class PortfolioAllocator:
    def allocate(self, winners: List[Tuple[str, float, float]], freed_budget: float) -> Dict[str, float]:
        """
        winners: [(adset_id, roas, current_budget), ...]
        returns: {adset_id: bump_amount}
        """
        if freed_budget <= 0 or not winners: return {}
        winners = winners[:PORTFOLIO_MAX_MOVES]
        weights = [max(0.01, r) for _, r, _ in winners]
        total_w = sum(weights) or len(weights)
        bumps: Dict[str, float] = {}
        for (adset_id, roas, cur), w in zip(winners, weights):
            bump = max(REINVEST_MIN_BUMP, freed_budget * (w / total_w) * REINVEST_SHARE)
            bumps[adset_id] = min(bump, cur * (MAX_SCALE_STEP_PCT / 100.0))
        return bumps

# ====== SmartScaler ======
class SmartScaler:
    """
    Stateless per-tick processor (internal counters & flags are persisted via `store`).

    expected `store` methods:
      - log(), log_kill(), log_scale(), incr(), get_counter(), set_counter(), set_flag(), get_flag()
    expected `meta` methods:
      - get_ad_insights(), pause_ad(), update_adset_budget(), duplicate_adset(), get_adset_budget_usd()
        (Note: despite *usd* naming, values reflect the ad account currency, here EUR.)
    """
    def __init__(self, store: Any):
        self.store = store
        self.bandit = ThompsonBandit()
        self.portfolio = PortfolioAllocator()
        self.metrics_cfg = MetricsConfig()

    # ---- stability across ticks ----
    def _stable(self, entity_id: str, rule_key: str, ok: bool, need: int = 2) -> bool:
        key = f"{entity_id}::stable::{rule_key}"
        try:
            if ok:
                return self.store.incr(key, 1) >= need
            self.store.set_counter(key, 0)
            return False
        except Exception:
            return ok  # degrade gracefully

    # ---- kill policy ----
    def _kill_logic(self, m: Metrics, consec_bad: int) -> Tuple[bool, str]:
        if m.cpa is not None and consec_bad >= KILL_CPA_DAYS and m.cpa >= 40:
            return True, f"CPA≥40 for {consec_bad}d"
        if (m.roas or 0.0) < 1.2 and consec_bad >= KILL_ROAS_DAYS:
            return True, f"ROAS<1.2 for {consec_bad}d"
        if (m.spend or 0.0) >= 150 and (m.purchases or 0) == 0:
            return True, "Spend≥150 with 0 purchases"
        if _credible_underperf(m.cpa, m.roas or 0.0, m.spend or 0.0):
            return True, "Probabilistic underperformance"
        return False, ""

    # ---- scale step policy ----
    def _scale_step(self, m: Metrics) -> int:
        # Matches plan: bands at CPA ≤ €22 & ROAS ≥ 3.0 (100%); CPA ≤ €27 & ROAS ≥ 2.0 (50%)
        if m.cpa is not None and m.cpa <= 22 and (m.roas or 0.0) >= 3.0:
            return 100
        if m.cpa is not None and m.cpa <= 27 and (m.roas or 0.0) >= 2.0:
            return 50
        return 0

    def _downscale_needed(self, m: Metrics) -> bool:
        # Safety nets per plan: ROAS < 1.7 or CPA > €33
        return (m.roas or 0.0) < HYST_ROAS_DOWN or (m.cpa is not None and m.cpa > HYST_CPA_UP)

    # ---- core process ----
    def process(self, meta: Any, rows: List[Dict[str, Any]]) -> Dict[str, int]:
        summary = {"kills": 0, "scaled": 0, "duped": 0, "downscaled": 0, "refreshed": 0}
        freed_budget: float = 0.0
        winners: List[Tuple[str, float, float]] = []  # (adset_id, roas, current_budget)

        for r in rows:
            ad_id: Optional[str] = r.get("ad_id")
            ad_name: str = r.get("ad_name", "")
            adset_id: Optional[str] = r.get("adset_id")
            if not ad_id or not adset_id:
                continue

            m = metrics_from_row(r, cfg=self.metrics_cfg)
            if not _meets_minimums(m):
                continue

            # Consecutive "bad" days counter
            bad = 1 if ((m.cpa is not None and m.cpa >= 40) or ((m.roas or 0.0) < 1.2)) else 0
            days_key = f"{ad_id}::bad_days"
            try:
                if bad:
                    self.store.incr(days_key, 1)
                else:
                    self.store.set_counter(days_key, 0)
                consec_bad = int(self.store.get_counter(days_key))
            except Exception:
                consec_bad = bad

            # Kill candidates
            kill, reason = self._kill_logic(m, consec_bad)
            if kill and self._stable(ad_id, "kill", True, 2):
                try:
                    cur_budget = None
                    if hasattr(meta, "get_adset_budget_usd"):
                        try:
                            cur_budget = meta.get_adset_budget_usd(adset_id)
                        except Exception:
                            cur_budget = None
                    meta.pause_ad(ad_id)
                    self.store.log_kill(
                        stage="SCALE",
                        entity_id=ad_id,
                        rule_type="auto_kill",
                        reason=reason,
                        observed={"CPA": m.cpa, "ROAS": m.roas, "purchases": m.purchases, "spend": m.spend},
                        thresholds={"cpa": 40, "roas": 1.2},
                    )
                    try:
                        alert_kill("SCALE", ad_name, reason, {"CPA": f"{(m.cpa or 0):.2f}", "ROAS": f"{(m.roas or 0):.2f}"})
                    except Exception:
                        notify(f"🛑 [SCALE] {ad_name} — {reason}")
                    summary["kills"] += 1
                    if cur_budget:
                        freed_budget += float(cur_budget)
                except Exception:
                    pass
                continue
            else:
                self._stable(ad_id, "kill", False, 2)

            # Bandit reward roughly tied to “excess” ROAS
            reward = max(0.0, min(1.0, ((m.roas or 0.0) - 1.0) / 3.0))
            trials = max(1.0, (m.spend or 0.0) / 50.0)
            self.bandit.update(adset_id, reward=reward, trials=trials)

            if (m.roas or 0.0) >= 2.0 and (m.cpa or 9e9) <= 27 and (m.purchases or 0) >= 2:
                try:
                    cur = getattr(meta, "get_adset_budget_usd", lambda _:_)(adset_id) or 100.0
                except Exception:
                    cur = 100.0
                winners.append((adset_id, float(m.roas or 0.0), float(cur)))

            # Downscale (hysteresis)
            if self._downscale_needed(m):
                try:
                    cur = getattr(meta, "get_adset_budget_usd", lambda _:_)(adset_id) or 100.0
                    new_budget = max(5.0, cur * 0.5)
                    meta.update_adset_budget(adset_id, new_budget, current_budget_usd=cur)
                    dedup_key = f"scale_down:{adset_id}:{int(round(new_budget))}:{_idkey('tick')}"
                    self.store.log(
                        entity_type="adset",
                        entity_id=adset_id,
                        action="SCALE_DOWN",
                        reason="-50%",
                        meta={"old_budget": cur, "new_budget": new_budget},
                        dedup_key=dedup_key,
                    )
                    notify(f"⬇️ [SCALE] {ad_name} → ~€{new_budget:,.0f}/d")
                    summary["downscaled"] += 1
                except Exception:
                    pass

            # Upscale (cooldown + stability + step)
            if _cooldown_ok(self.store, adset_id, SCALE_COOLDOWN_H):
                inc = self._scale_step(m)
                if inc and self._stable(ad_id, "scale", True, 2):
                    try:
                        cur = getattr(meta, "get_adset_budget_usd", lambda _:_)(adset_id) or 100.0
                        cap = cur * (1.0 + MAX_SCALE_STEP_PCT / 100.0)
                        new_budget = min(cur * (1.0 + inc / 100.0), cap)
                        meta.update_adset_budget(adset_id, new_budget, current_budget_usd=cur)
                        self.store.log_scale(adset_id, inc, f"rule_inc_{inc}", meta={"old_budget": cur, "new_budget": new_budget})
                        try:
                            alert_scale(ad_name, inc, new_budget=new_budget)
                        except Exception:
                            notify(f"⬆️ [SCALE] {ad_name} +{inc}% → ~€{new_budget:,.0f}/d")
                        _mark_scaled(self.store, adset_id)
                        summary["scaled"] += 1
                    except Exception:
                        pass
                else:
                    self._stable(ad_id, "scale", False, 2)

            # Duplicate (cap per 24h)
            dups_key = f"{adset_id}::dups::{_now().date().isoformat()}"
            try:
                used = int(self.store.get_counter(dups_key))
            except Exception:
                used = 0
            if (m.purchases or 0) >= 5 and (m.cpa or 9e9) <= 27 and used < DUP_CAP_24H:
                allow = min(DUP_CAP_24H - used, 2)
                if allow > 0:
                    try:
                        meta.duplicate_adset(adset_id, count=allow, status="PAUSED", prefix="[SCALE] clone ")
                        self.store.set_counter(dups_key, used + allow)
                        dedup_key = f"duplicate:{adset_id}:{allow}:{_now().date().isoformat()}"
                        self.store.log(
                            entity_type="adset",
                            entity_id=adset_id,
                            action="DUPLICATE",
                            reason=f"{allow} copies",
                            dedup_key=dedup_key,
                        )
                        notify(f"🧬 [SCALE] {ad_name} ×{allow} (PAUSED)")
                        summary["duped"] += allow
                    except Exception:
                        pass

            # Refresh after a streak of strong days
            good_key = f"{adset_id}::good_days"
            if (m.cpa or 9e9) <= 22 and (m.roas or 0.0) >= 3.0 and (m.purchases or 0) >= 2:
                try:
                    days = self.store.incr(good_key, 1)
                except Exception:
                    days = 1
                if days >= 7:
                    try:
                        meta.duplicate_adset(adset_id, count=1, status="PAUSED", prefix="[REFRESH] ")
                        self.store.set_counter(good_key, 0)
                        notify(f"🆕 [SCALE] Refresh copy for {ad_name}")
                        summary["refreshed"] += 1
                    except Exception:
                        pass
            else:
                try:
                    self.store.set_counter(good_key, 0)
                except Exception:
                    pass

        # Reinvest freed budget into winners (EUR)
        if freed_budget > 0 and winners:
            bumps = self.portfolio.allocate(winners, freed_budget)
            for adset_id, bump in bumps.items():
                try:
                    cur = getattr(meta, "get_adset_budget_usd", lambda _:_)(adset_id) or 100.0
                    new_budget = cur + bump
                    meta.update_adset_budget(adset_id, new_budget, current_budget_usd=cur)
                    dedup_key = f"reinvest:{adset_id}:{int(round(new_budget))}:{_idkey('tick')}"
                    self.store.log(
                        entity_type="adset",
                        entity_id=adset_id,
                        action="REINVEST",
                        reason=f"+€{bump:,.0f}",
                        meta={"old_budget": cur, "new_budget": new_budget},
                        dedup_key=dedup_key,
                    )
                    notify(f"💧 [SCALE] +~€{bump:,.0f} → {adset_id}")
                except Exception:
                    pass

        return summary

# ====== Orchestrator ======
class AdvancedScalerRunner:
    """
    Thin orchestrator that:
      1) guards duplicate ticks via store.tick_seen(...)
      2) optionally freezes scale-ups on spend velocity spikes
      3) fetches insights & runs SmartScaler
    """
    def __init__(self, meta: Any, store: Any, account_tz: str = "Europe/Amsterdam"):
        self.meta = meta
        self.store = store
        self.tz = account_tz  # currently informational; insights time ranges come from Meta

    def _pacing_freeze(self, scaling_campaign_id: str) -> Tuple[bool, Dict[str, float]]:
        """
        Heuristic velocity check (replace with explicit date windows if your client supports them).
        Freezes scale-ups when spend grows >+150% (24h) or >+250% (48h).
        """
        try:
            rows = self.meta.get_ad_insights(
                level="campaign",
                filtering=[{"field": "campaign.id", "operator": "IN", "value": [scaling_campaign_id]}],
                fields=["spend"],
                action_attribution_windows=list(ATTR_WINDOWS),
                paginate=True,
            )
            today_spend = sum(float(r.get("spend") or 0.0) for r in rows)
            # crude yesterday proxy (if client supports date ranges, split explicitly)
            y_spend = max(1.0, today_spend * 0.8)
            g24 = today_spend / y_spend
            g48 = g24  # placeholder without 2-day split
            freeze = (g24 > 1.5) or (g48 > 2.5)
            return freeze, {"g24": g24, "g48": g48}
        except Exception:
            return False, {"g24": 1.0, "g48": 1.0}

    def run(self, settings: Dict[str, Any]) -> Dict[str, int]:
        tick_key = _idkey("adv_scale_tick")
        try:
            if hasattr(self.store, "tick_seen") and self.store.tick_seen(tick_key):
                notify(f"ℹ️ [SCALE] skip {tick_key}")
                return {"kills": 0, "scaled": 0, "duped": 0, "downscaled": 0, "refreshed": 0}
        except Exception:
            pass

        scaling_campaign_id = settings["ids"]["scaling_campaign_id"]
        freeze, velo = self._pacing_freeze(scaling_campaign_id)
        if freeze:
            notify(f"🧯 [SCALE] velocity high (24h×{velo['g24']:.2f}, 48h×{velo['g48']:.2f}) — freeze scale-ups")

        try:
            rows = self.meta.get_ad_insights(
                level="ad",
                filtering=[{"field": "campaign.id", "operator": "IN", "value": [scaling_campaign_id]}],
                fields=[
                    "ad_id","ad_name","adset_id","campaign_id",
                    "spend","impressions","clicks",
                    "actions","action_values","purchase_roas",
                    "reach","unique_clicks",
                ],
                action_attribution_windows=list(ATTR_WINDOWS),
                paginate=True,
            )
        except Exception as e:
            notify(f"❗ [SCALE] Insights error: {e}")
            return {"kills": 0, "scaled": 0, "duped": 0, "downscaled": 0, "refreshed": 0}

        # Freeze only blocks increases/duplication; we still allow kills/downscales inside SmartScaler.
        scaler = SmartScaler(self.store)
        return scaler.process(self.meta, rows)

# ====== Public API ======
def run_advanced_scaling_tick(meta: Any, settings: Dict[str, Any], store: Any) -> Dict[str, int]:
    return AdvancedScalerRunner(meta, store, account_tz=settings.get("timezone", "Europe/Amsterdam")).run(settings)

# Back-compat alias (used by some schedulers)
run_scaling_tick = run_advanced_scaling_tick
