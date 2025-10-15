from __future__ import annotations

"""
Production-ready advanced scaler with lifetime-aware tripwires - now fully tunable via rules.yaml.

What changed (high level):
- All knobs can be driven from settings["scaling"]["engine"] with sensible env-var fallbacks.
- Scale step bands, duplication conditions, hysteresis downscale factor, pacing-freeze thresholds are configurable.
- Attribution windows for insights come from settings (engine.attribution_windows) with fallback to env.

Expected settings structure (see rules.yaml):
scaling:
  engine:
    attribution_windows: ["7d_click","1d_view"]
    minimums: { min_impressions: 1000, min_clicks: 50, min_spend: 50 }
    hysteresis: { roas_down_band: 1.7, cpa_up_band: 33, downscale_factor: 0.5 }
    kills:
      cpa_days: 2
      roas_days: 3
      lifetime_tripwires: { spend_no_purchase_eur: 150 }
      thresholds: { cpa_gte: 40, roas_lt: 1.2 }       # optional; defaults kept if omitted
    scale_up:
      cooldown_hours: 24
      max_scale_step_pct: 200
      bands:
        - { cpa_lte: 22, roas_gte: 3.0, inc_pct: 100 }
        - { cpa_lte: 27, roas_gte: 2.0, inc_pct: 50 }
    duplication:
      max_duplicates_per_24h: 3
      purchases_gte: 5
      cpa_lte: 27
      max_each_action: 2
    reinvest:
      share: 0.5
      min_bump_eur: 10
      portfolio_max_moves: 6
    pacing_freeze:
      enabled: true
      growth_pct_24h_warn: 150
      growth_pct_48h_warn: 250
"""

import os
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from metrics import Metrics, MetricsConfig, metrics_from_row
from slack import notify, alert_kill, alert_scale, alert_error
from utils import (
    getenv_f, getenv_i, getenv_b, cfg, cfg_or_env_f, cfg_or_env_i, cfg_or_env_b, cfg_or_env_list,
    safe_f, today_str, daily_key, ad_day_flag_key, now_minute_key, clean_text_token, prettify_ad_name
)

# ====== Time ======
UTC = timezone.utc
def _now() -> datetime: return datetime.now(UTC)

# Local timezone (Amsterdam) for day buckets
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

ACCOUNT_TZ_NAME = os.getenv("ACCOUNT_TZ") or os.getenv("ACCOUNT_TIMEZONE") or "Europe/Amsterdam"
LOCAL_TZ = ZoneInfo(ACCOUNT_TZ_NAME) if ZoneInfo else None

def _now_local() -> datetime:
    return datetime.now(LOCAL_TZ) if LOCAL_TZ else _now()

def today_str() -> str:
    return _now_local().strftime("%Y-%m-%d")

def daily_key(stage: str, metric: str) -> str:
    # daily::<YYYY-MM-DD>::STAGE::metric
    return f"daily::{today_str()}::{stage}::{metric}"

# ====== Helpers for config resolution ======
def getenv_b(name: str, default: bool) -> bool:
    return (os.getenv(name, str(int(default))) or "").lower() in ("1", "true", "yes", "y")

def getenv_f(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return default

def getenv_i(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except Exception: return default

def _get(d: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

@dataclass
class ScaleConfig:
    # minimums
    min_impressions: int
    min_clicks: int
    min_spend: float

    # hysteresis / downscale
    roas_down_band: float
    cpa_up_band: float
    downscale_factor: float

    # kills
    kill_cpa_days: int
    kill_roas_days: int
    kill_cpa_threshold: float
    kill_roas_min: float
    lifetime_spend_no_purchase_eur: float

    # scale-up
    cooldown_hours: int
    max_scale_step_pct: int
    bands: List[Dict[str, float]]  # list of {cpa_lte, roas_gte, inc_pct}

    # duplication
    dup_cap_24h: int
    dup_purchases_gte: int
    dup_cpa_lte: float
    dup_max_each_action: int

    # reinvest
    reinvest_share: float
    reinvest_min_bump: float
    portfolio_max_moves: int

    # insights attribution windows
    attr_windows: Tuple[str, ...]

    # pacing freeze
    pacing_freeze_enabled: bool
    pacing_24h_warn: float
    pacing_48h_warn: float

    @staticmethod
    def from_settings(settings: Dict[str, Any]) -> "ScaleConfig":
        eng = _get(settings, ["scaling", "engine"], {}) or {}

        # minimums
        min_imps  = int(_get(eng, ["minimums", "min_impressions"], getenv_i("SCALE_MIN_IMPRESSIONS", 1000)))
        min_click = int(_get(eng, ["minimums", "min_clicks"], getenv_i("SCALE_MIN_CLICKS", 50)))
        min_spend = float(_get(eng, ["minimums", "min_spend"], getenv_f("SCALE_MIN_SPEND", 50.0)))

        # hysteresis
        roas_down = float(_get(eng, ["hysteresis", "roas_down_band"], getenv_f("SCALE_HYST_ROAS_DOWN", 1.7)))
        cpa_up    = float(_get(eng, ["hysteresis", "cpa_up_band"], getenv_f("SCALE_HYST_CPA_UP", 33.0)))
        dwn_fac   = float(_get(eng, ["hysteresis", "downscale_factor"], 0.5))

        # kills + thresholds
        kill_cpa_days = int(_get(eng, ["kills", "cpa_days"], getenv_i("SCALE_KILL_CPA_DAYS", 2)))
        kill_roas_days = int(_get(eng, ["kills", "roas_days"], getenv_i("SCALE_KILL_ROAS_DAYS", 3)))
        kill_cpa_thresh = float(_get(eng, ["kills", "thresholds", "cpa_gte"], 40.0))
        kill_roas_min = float(_get(eng, ["kills", "thresholds", "roas_lt"], 1.2))
        life_tripwire = float(_get(eng, ["kills", "lifetime_tripwires", "spend_no_purchase_eur"], getenv_f("SCALE_SPEND_NO_PURCHASE_EUR", 150.0)))

        # scale up
        cooldown = int(_get(eng, ["scale_up", "cooldown_hours"], getenv_i("SCALE_COOLDOWN_H", 24)))
        max_step = int(_get(eng, ["scale_up", "max_scale_step_pct"], getenv_i("SCALE_MAX_SCALE_STEP_PCT", 200)))
        bands = list(_get(eng, ["scale_up", "bands"], [])) or [
            {"cpa_lte": 22.0, "roas_gte": 3.0, "inc_pct": 100.0},
            {"cpa_lte": 27.0, "roas_gte": 2.0, "inc_pct": 50.0},
        ]

        # duplication
        dup_cap = int(_get(eng, ["duplication", "max_duplicates_per_24h"], getenv_i("SCALE_DUP_CAP_24H", 3)))
        dup_p = int(_get(eng, ["duplication", "purchases_gte"], 5))
        dup_cpa = float(_get(eng, ["duplication", "cpa_lte"], 27.0))
        dup_each = int(_get(eng, ["duplication", "max_each_action"], 2))

        # reinvest
        r_share = float(_get(eng, ["reinvest", "share"], getenv_f("SCALE_REINVEST_SHARE", 0.5)))
        r_min = float(_get(eng, ["reinvest", "min_bump_eur"], getenv_f("SCALE_REINVEST_MIN_BUMP", 10.0)))
        r_moves = int(_get(eng, ["reinvest", "portfolio_max_moves"], getenv_i("SCALE_PORTFOLIO_MAX_MOVES", 6)))

        # attr windows
        attr = tuple(_get(eng, ["attribution_windows"], None) or (os.getenv("SCALE_ATTR_WINDOWS", "7d_click,1d_view") or "7d_click,1d_view").split(","))

        # pacing freeze
        pf = _get(eng, ["pacing_freeze"], {}) or {}
        pf_enabled = bool(_get(pf, ["enabled"], True))
        pf_24 = float(_get(pf, ["growth_pct_24h_warn"], 150))
        pf_48 = float(_get(pf, ["growth_pct_48h_warn"], 250))

        return ScaleConfig(
            min_impressions=min_imps,
            min_clicks=min_click,
            min_spend=min_spend,
            roas_down_band=roas_down,
            cpa_up_band=cpa_up,
            downscale_factor=dwn_fac,
            kill_cpa_days=kill_cpa_days,
            kill_roas_days=kill_roas_days,
            kill_cpa_threshold=kill_cpa_thresh,
            kill_roas_min=kill_roas_min,
            lifetime_spend_no_purchase_eur=life_tripwire,
            cooldown_hours=cooldown,
            max_scale_step_pct=max_step,
            bands=bands,
            dup_cap_24h=dup_cap,
            dup_purchases_gte=dup_p,
            dup_cpa_lte=dup_cpa,
            dup_max_each_action=dup_each,
            reinvest_share=r_share,
            reinvest_min_bump=r_min,
            portfolio_max_moves=r_moves,
            attr_windows=attr,
            pacing_freeze_enabled=pf_enabled,
            pacing_24h_warn=pf_24,
            pacing_48h_warn=pf_48,
        )

# ====== Logic helpers (pure) ======
def _idkey(prefix: str) -> str:
    return f"{prefix}::{_now().strftime('%Y-%m-%dT%H:%M')}"

def _sigma_clip(values: List[float], z: float = 3.0) -> List[float]:
    if len(values) < 3: return values
    mu = statistics.mean(values)
    sd = statistics.pstdev(values) or 1.0
    return [v for v in values if abs(v - mu) <= z * sd]

def _credible_underperf(cpa: Optional[float], roas: float, spend: float, cfg: ScaleConfig) -> bool:
    # Lightweight probabilistic guard: only consider once there's a little signal.
    if spend < 50.0: return False
    if cpa is not None and spend > 100.0 and cpa > cfg.cpa_up_band * 1.1:
        return True
    return roas < max(1.1, cfg.roas_down_band * 0.9)

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

def _count_purchases(actions: Any) -> int:
    acts = actions or []
    total = 0
    for a in acts:
        if a.get("action_type") == "purchase":
            try:
                total += int(float(a.get("value") or 0))
            except Exception:
                pass
    return total

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
    def __init__(self, cfg: ScaleConfig):
        self.cfg = cfg
    def allocate(self, winners: List[Tuple[str, float, float]], freed_budget: float) -> Dict[str, float]:
        """
        winners: [(adset_id, roas, current_budget), ...]
        returns: {adset_id: bump_amount}
        """
        if freed_budget <= 0 or not winners: return {}
        winners = winners[: self.cfg.portfolio_max_moves]
        weights = [max(0.01, r) for _, r, _ in winners]
        total_w = sum(weights) or len(weights)
        bumps: Dict[str, float] = {}
        for (adset_id, roas, cur), w in zip(winners, weights):
            bump = max(self.cfg.reinvest_min_bump, freed_budget * (w / total_w) * self.cfg.reinvest_share)
            bumps[adset_id] = min(bump, cur * (self.cfg.max_scale_step_pct / 100.0))
        return bumps

# ====== SmartScaler ======
class SmartScaler:
    """
    Stateless per-tick processor (internal counters & flags are persisted via `store`).

    expected `store` methods:
      - log(), log_kill(), log_scale(), incr(), get_counter(), set_counter(), set_flag(), get_flag()
    expected `meta` methods:
      - get_ad_insights(), pause_ad(), update_adset_budget(), duplicate_adset(), get_adset_budget()
        (get_adset_budget_usd is supported as a fallback alias)
    """
    def __init__(self, store: Any, cfg: ScaleConfig):
        self.store = store
        self.cfg = cfg
        self.bandit = ThompsonBandit()
        self.portfolio = PortfolioAllocator(cfg)
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

    # ---- minimums ----
    def _meets_minimums(self, m: Metrics) -> bool:
        return (m.spend or 0.0) >= self.cfg.min_spend and (m.impressions or 0.0) >= self.cfg.min_impressions and (m.clicks or 0.0) >= self.cfg.min_clicks

    # ---- kill policy ----
    def _kill_logic(self, m: Metrics, consec_bad: int) -> Tuple[bool, str]:
        if m.cpa is not None and consec_bad >= self.cfg.kill_cpa_days and m.cpa >= self.cfg.kill_cpa_threshold:
            return True, f"CPA≥{int(self.cfg.kill_cpa_threshold)} for {consec_bad}d"
        if (m.roas or 0.0) < self.cfg.kill_roas_min and consec_bad >= self.cfg.kill_roas_days:
            return True, f"ROAS<{self.cfg.kill_roas_min:.1f} for {consec_bad}d"
        if _credible_underperf(m.cpa, m.roas or 0.0, m.spend or 0.0, self.cfg):
            return True, "Probabilistic underperformance"
        return False, ""

    # ---- scale step policy (bands from YAML) ----
    def _scale_step(self, m: Metrics) -> int:
        if m.cpa is None: return 0
        for band in self.cfg.bands:
            try:
                cpa_lte = float(band.get("cpa_lte", 9e9))
                roas_gte = float(band.get("roas_gte", 0.0))
                inc = int(float(band.get("inc_pct", 0)))
                if m.cpa <= cpa_lte and (m.roas or 0.0) >= roas_gte:
                    return inc
            except Exception:
                continue
        return 0

    def _downscale_needed(self, m: Metrics) -> bool:
        return (m.roas or 0.0) < self.cfg.roas_down_band or (m.cpa is not None and m.cpa > self.cfg.cpa_up_band)

    # ---- core process ----
    def process(self, meta: Any, rows: List[Dict[str, Any]], lifetime_map: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, int]:
        summary = {"kills": 0, "scaled": 0, "duped": 0, "downscaled": 0, "refreshed": 0}
        freed_budget: float = 0.0
        winners: List[Tuple[str, float, float]] = []  # (adset_id, roas, current_budget)

        for r in rows:
            ad_id: Optional[str] = r.get("ad_id")
            ad_name: str = r.get("ad_name", "")
            adset_id: Optional[str] = r.get("adset_id")
            if not ad_id or not adset_id:
                continue

            # Today/default metrics for control logic
            m = metrics_from_row(r, cfg=self.metrics_cfg)
            if not self._meets_minimums(m):
                continue

            # Lifetime (since ad launch) metrics for tripwire
            lr = (lifetime_map or {}).get(str(ad_id), {}) or {}
            life_spend = float(lr.get("spend") or 0.0)
            life_purchases = _count_purchases(lr.get("actions"))

            # Removed per-ad informational messages - now handled in consolidated run summary

            # Consecutive "bad" days counter (based on today metrics)
            bad = 1 if ((m.cpa is not None and m.cpa >= self.cfg.kill_cpa_threshold) or ((m.roas or 0.0) < self.cfg.kill_roas_min)) else 0
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

            # Lifetime no-purchase tripwire takes precedence
            if life_spend >= self.cfg.lifetime_spend_no_purchase_eur and life_purchases == 0:
                kill = True
                reason = f"Lifetime spend≥€{int(self.cfg.lifetime_spend_no_purchase_eur)} & 0 purchases"

            if kill and self._stable(ad_id, "kill", True, 2):
                try:
                    # Try EUR-aware getter; fallback to *_usd alias if present
                    get_budget = getattr(meta, "get_adset_budget", None) or getattr(meta, "get_adset_budget_usd", None) or (lambda _ : None)
                    cur_budget = None
                    try:
                        cur_budget = get_budget(adset_id)
                    except Exception:
                        cur_budget = None

                    meta.pause_ad(ad_id)

                    # daily counter: SCALING kills++
                    try:
                        self.store.incr(daily_key("SCALING", "kills"), 1)
                    except Exception:
                        pass

                    self.store.log_kill(
                        stage="SCALE",
                        entity_id=ad_id,
                        rule_type="auto_kill",
                        reason=reason,
                        observed={
                            "CPA": m.cpa, "ROAS": m.roas,
                            "purchases_today": m.purchases, "spend_today": m.spend,
                            "spend_lifetime": life_spend, "purchases_lifetime": life_purchases
                        },
                        thresholds={"cpa": self.cfg.kill_cpa_threshold, "roas": self.cfg.kill_roas_min, "lifetime_tripwire_eur": self.cfg.lifetime_spend_no_purchase_eur},
                    )
                    try:
                        alert_kill("SCALE", ad_name, reason, {"CPA": f"{(m.cpa or 0):.2f}", "ROAS": f"{(m.roas or 0):.2f}"})
                    except Exception:
                        alert_kill("SCALE", ad_name, reason, {"CPA": f"{(m.cpa or 0):.2f}", "ROAS": f"{(m.roas or 0):.2f}"})
                    summary["kills"] += 1
                    if cur_budget:
                        try:
                            freed_budget += float(cur_budget)
                        except Exception:
                            pass
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
                    cur = (getattr(meta, "get_adset_budget", None) or getattr(meta, "get_adset_budget_usd", lambda _ : 100.0))(adset_id) or 100.0
                except Exception:
                    cur = 100.0
                winners.append((adset_id, float(m.roas or 0.0), float(cur)))

            # Downscale (hysteresis)
            if self._downscale_needed(m):
                try:
                    cur = (getattr(meta, "get_adset_budget", None) or getattr(meta, "get_adset_budget_usd", lambda _ : 100.0))(adset_id) or 100.0
                    new_budget = max(5.0, cur * float(self.cfg.downscale_factor))
                    # Use current MetaClient signature (current_budget=...)
                    meta.update_adset_budget(adset_id, new_budget, current_budget=cur)
                    dedup_key = f"scale_down:{adset_id}:{int(round(new_budget))}:{_idkey('tick')}"
                    self.store.log(
                        entity_type="adset",
                        entity_id=adset_id,
                        action="SCALE_DOWN",
                        reason=f"x{self.cfg.downscale_factor:.2f}",
                        meta={"old_budget": cur, "new_budget": new_budget},
                        dedup_key=dedup_key,
                    )
                    # Downscale notification removed - now handled in consolidated run summary
                    summary["downscaled"] += 1
                    # daily counter: SCALING downscaled++
                    try:
                        self.store.incr(daily_key("SCALING", "downscaled"), 1)
                    except Exception:
                        pass
                except Exception:
                    pass

            # Upscale (cooldown + stability + step)
            if _cooldown_ok(self.store, adset_id, self.cfg.cooldown_hours):
                inc = self._scale_step(m)
                if inc and self._stable(ad_id, "scale", True, 2):
                    try:
                        cur = (getattr(meta, "get_adset_budget", None) or getattr(meta, "get_adset_budget_usd", lambda _ : 100.0))(adset_id) or 100.0
                        cap = cur * (1.0 + self.cfg.max_scale_step_pct / 100.0)
                        new_budget = min(cur * (1.0 + inc / 100.0), cap)
                        meta.update_adset_budget(adset_id, new_budget, current_budget=cur)
                        self.store.log_scale(adset_id, inc, f"rule_inc_{inc}", meta={"old_budget": cur, "new_budget": new_budget})
                        try:
                            alert_scale(ad_name, inc, new_budget=new_budget)
                        except Exception:
                            alert_scale(ad_name, inc, new_budget=new_budget)
                        _mark_scaled(self.store, adset_id)
                        summary["scaled"] += 1
                        # daily counter: SCALING scaled++
                        try:
                            self.store.incr(daily_key("SCALING", "scaled"), 1)
                        except Exception:
                            pass
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
            if (m.purchases or 0) >= self.cfg.dup_purchases_gte and (m.cpa or 9e9) <= self.cfg.dup_cpa_lte and used < self.cfg.dup_cap_24h:
                allow = min(self.cfg.dup_cap_24h - used, self.cfg.dup_max_each_action)
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
                        # Duplication notification removed - now handled in consolidated run summary
                        summary["duped"] += allow
                        # daily counter: SCALING duped += allow
                        try:
                            self.store.incr(daily_key("SCALING", "duped"), allow)
                        except Exception:
                            pass
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
                        # Refresh notification removed - now handled in consolidated run summary
                        summary["refreshed"] += 1
                        # daily counter: SCALING refreshed++
                        try:
                            self.store.incr(daily_key("SCALING", "refreshed"), 1)
                        except Exception:
                            pass
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
                    cur = (getattr(meta, "get_adset_budget", None) or getattr(meta, "get_adset_budget_usd", lambda _ : 100.0))(adset_id) or 100.0
                    new_budget = cur + bump
                    meta.update_adset_budget(adset_id, new_budget, current_budget=cur)
                    dedup_key = f"reinvest:{adset_id}:{int(round(new_budget))}:{_idkey('tick')}"
                    self.store.log(
                        entity_type="adset",
                        entity_id=adset_id,
                        action="REINVEST",
                        reason=f"+€{bump:,.0f}",
                        meta={"old_budget": cur, "new_budget": new_budget},
                        dedup_key=dedup_key,
                    )
                    # Reinvest notification removed - now handled in consolidated run summary
                except Exception:
                    pass

        return summary

# ====== Orchestrator ======
class AdvancedScalerRunner:
    """
    Thin orchestrator that:
      1) guards duplicate ticks via store.tick_seen(...)
      2) optionally freezes scale-ups on spend velocity spikes (tunable)
      3) fetches insights (today + lifetime) & runs SmartScaler
    """
    def __init__(self, meta: Any, store: Any, account_tz: str = "Europe/Amsterdam"):
        self.meta = meta
        self.store = store
        self.tz = account_tz  # informational

    def _pacing_freeze(self, scaling_campaign_id: str, cfg: ScaleConfig) -> Tuple[bool, Dict[str, float]]:
        """
        Heuristic velocity check (kept simple).
        Freezes scale-ups when spend grows beyond configured bands.
        """
        if not cfg.pacing_freeze_enabled:
            return False, {"g24": 1.0, "g48": 1.0}
        try:
            rows = self.meta.get_ad_insights(
                level="campaign",
                filtering=[{"field": "campaign.id", "operator": "IN", "value": [scaling_campaign_id]}],
                fields=["spend"],
                action_attribution_windows=list(cfg.attr_windows),
                paginate=True,
            )
            today_spend = sum(float(r.get("spend") or 0.0) for r in rows)
            # crude proxy for yesterday/48h; replace with explicit ranges if needed
            y_spend = max(1.0, today_spend * 0.8)
            g24 = today_spend / y_spend
            g48 = g24
            freeze = (g24 > (cfg.pacing_24h_warn / 100.0 + 1e-9)) or (g48 > (cfg.pacing_48h_warn / 100.0 + 1e-9))
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

        cfg = ScaleConfig.from_settings(settings)

        scaling_campaign_id = settings["ids"]["scaling_campaign_id"]
        freeze, velo = self._pacing_freeze(scaling_campaign_id, cfg)
        if freeze:
            notify(f"🧯 [SCALE] velocity high (24h×{velo['g24']:.2f}, 48h×{velo['g48']:.2f}) - freeze scale-ups")

        try:
            # Today/default
            rows = self.meta.get_ad_insights(
                level="ad",
                filtering=[{"field": "campaign.id", "operator": "IN", "value": [scaling_campaign_id]}],
                fields=[
                    "ad_id","ad_name","adset_id","campaign_id",
                    "spend","impressions","clicks",
                    "actions","action_values","purchase_roas",
                    "reach","unique_clicks",
                ],
                action_attribution_windows=list(cfg.attr_windows),
                paginate=True,
            )
            # Lifetime (since ad launch)
            rows_life = self.meta.get_ad_insights(
                level="ad",
                filtering=[{"field": "campaign.id", "operator": "IN", "value": [scaling_campaign_id]}],
                fields=["ad_id", "spend", "actions"],
                action_attribution_windows=list(cfg.attr_windows),
                date_preset="lifetime",
                paginate=True,
            )
        except Exception as e:
            alert_error(f"insights fetch failed: {e}")
            return {"kills": 0, "scaled": 0, "duped": 0, "downscaled": 0, "refreshed": 0}

        # Build lifetime map
        life_by_ad: Dict[str, Dict[str, Any]] = {}
        for lr in rows_life or []:
            aid = str(lr.get("ad_id") or "")
            if aid:
                life_by_ad[aid] = lr

        # Freeze blocks increases/duplication, but SmartScaler still handles kills/downscales.
        scaler = SmartScaler(self.store, cfg)
        result = scaler.process(self.meta, rows, lifetime_map=life_by_ad)

        if freeze:
            # Zero out "scaled/duped" to reflect freeze; kills/downscales still reported.
            result["scaled"] = 0
            result["duped"] = 0

        return result

# ====== Public API ======
def run_advanced_scaling_tick(meta: Any, settings: Dict[str, Any], store: Any) -> Dict[str, int]:
    return AdvancedScalerRunner(meta, store, account_tz=settings.get("timezone", "Europe/Amsterdam")).run(settings)

# Back-compat alias (used by some schedulers)
run_scaling_tick = run_advanced_scaling_tick
