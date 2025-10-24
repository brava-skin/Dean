# rules.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytz

from analytics.metrics import (
    MetricsConfig,
    Metrics,
    metrics_from_row,
    aggregate_rows,
    tripwire_threshold_account,
)

UTC = timezone.utc

# ---- Environment / defaults (EUR account; product price in USD) ----
PRODUCT_PRICE = float(os.getenv("PRODUCT_PRICE", "50") or 50)   # USD price of the product (used for tripwire)
BREAKEVEN_CPA = float(os.getenv("BREAKEVEN_CPA", "27.51") or 27.51)   # EUR (account currency)
COGS_ENV = os.getenv("COGS_PER_PURCHASE")
COGS_PER_PURCHASE = float(COGS_ENV) if COGS_ENV not in (None, "") else None


def _f(x: Any, d: float = 0.0) -> float:
    try:
        if x is None:
            return d
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).replace(",", "").strip())
    except Exception:
        return d


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _pct_growth(curr: float, prev: float) -> float:
    if prev <= 0:
        return 999.0 if curr > 0 else 0.0
    return (curr - prev) / prev * 100.0


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) or 1.0
    return [v / s for v in ex]


@dataclass
class AdaptiveBands:
    ctr_floor: float
    cpa_max: float
    roas_min: float


# ---- Compatibility shims to keep working with older metrics signatures ----
def _metrics_from_row_compat(
    row: Dict[str, Any],
    *,
    cfg: MetricsConfig,
    prefer_roas_field: bool,
    cogs_per_purchase: Optional[float],
    smoothing_epsilon: float,
) -> Metrics:
    try:
        return metrics_from_row(
            row,
            cfg=cfg,
            prefer_roas_field=prefer_roas_field,
            cogs_per_purchase=cogs_per_purchase,
            smoothing_epsilon=smoothing_epsilon,
        )  # type: ignore[call-arg]
    except TypeError:
        try:
            return metrics_from_row(
                row,
                cfg=cfg,
                cogs_per_purchase=cogs_per_purchase,
                smoothing_epsilon=smoothing_epsilon,
            )  # type: ignore[call-arg]
        except TypeError:
            try:
                return metrics_from_row(row, cfg=cfg, cogs_per_purchase=cogs_per_purchase)  # type: ignore[call-arg]
            except TypeError:
                return metrics_from_row(row, cfg=cfg)


def _aggregate_rows_compat(
    rows: List[Dict[str, Any]],
    *,
    cfg: MetricsConfig,
    smoothing_epsilon: float,
) -> Metrics:
    try:
        return aggregate_rows(rows, cfg=cfg, smoothing_epsilon=smoothing_epsilon)  # type: ignore[call-arg]
    except TypeError:
        return aggregate_rows(rows, cfg=cfg)


class AdvancedRuleEngine:
    """
    Production-ready rule engine for ad automation:
      - Metric computation with compatibility shims
      - Stage rules (testing, validation, scaling)
      - Stability (N consecutive ticks) and minimums
      - Adaptive account bands via EWMA
      - Guardrails (pacing, account CPA, tripwires)
      - Reinvestment allocator
      - Creative compliance
      - TZ-aware daily counters (Europe/Amsterdam by default)
      - Accessors for validation.engine and scaling.engine tunables (new)
    """

    def __init__(self, cfg: Dict[str, Any], store: Optional[Any] = None):
        self.cfg = cfg or {}
        self.store = store
        self._mem_counters: Dict[str, int] = {}
        self._mem_flags: Dict[Tuple[str, str, str], str] = {}

        # --- Modes
        self.mode = (self.cfg.get("mode", {}).get("current") or "production").lower()

        # --- Stability
        stab = self.cfg.get("stability", {}) or {}
        self.consecutive_ticks = int(stab.get("consecutive_ticks", 1) or 1)
        self.eps = float(stab.get("smoothing_epsilon", 0.0) or 0.0)

        # --- Minimums & attribution
        self.global_mins = self.cfg.get("minimums", {}) or {}
        attr = self.cfg.get("attribution", {}) or {}
        self.roas_source = (attr.get("roas_source") or "computed").lower()

        # --- Timezones (account vs audience). Account drives 'day' boundaries.
        tz_cfg = self.cfg.get("timezones", {}) or {}
        self.account_tz_name = tz_cfg.get("account_tz") or "Europe/Amsterdam"
        self.audience_tz_name = tz_cfg.get("audience_tz") or "America/Chicago"
        try:
            self.account_tz = pytz.timezone(self.account_tz_name)
        except Exception:
            self.account_tz = pytz.timezone("Europe/Amsterdam")

        # --- Metrics config (EUR account; USD product; pass FX rate if provided)
        fx_cfg = ((self.cfg.get("economics", {}) or {}).get("fx", {}) or {})
        # From env > YAML default_rate > fallback 0.92
        fx_rate_env = os.getenv(fx_cfg.get("env_rate_var", "USD_EUR_RATE") or "USD_EUR_RATE")
        usd_eur_rate = float(fx_rate_env) if fx_rate_env else float(fx_cfg.get("default_rate", 0.92))

        self.metrics_cfg = MetricsConfig(
            prefer_roas_field=(self.roas_source == "field"),
            account_currency=(self.cfg.get("economics", {}).get("account_currency") or "EUR"),
            product_currency=(fx_cfg.get("base_product_currency") or "USD"),
            usd_eur_rate=usd_eur_rate,
        )

        # --- COGS source
        econ = self.cfg.get("economics", {}) or {}
        self.use_env_cogs = bool(econ.get("use_env_cogs", True))
        self.use_creative_cogs = bool(econ.get("use_creative_cogs", False))

        # --- Creative compliance
        cc = self.cfg.get("creative_compliance") or {}
        self.cc_require_fields: List[str] = list(cc.get("require_fields") or [])
        self.cc_forbid_terms: List[str] = [str(t).lower() for t in (cc.get("forbid_terms") or [])]
        self.cc_max_lengths: Dict[str, int] = {k: int(v) for k, v in (cc.get("max_lengths") or {}).items()}

    # ---------- Time helpers ----------
    def _now_account(self) -> datetime:
        """Account-local 'now' (aware)."""
        return datetime.now(self.account_tz)

    def _today_key_account(self) -> str:
        """Date key in account-local time (for daily duplicate caps, etc.)."""
        return self._now_account().date().isoformat()

    # ---------- Store helpers ----------
    def _key(self, entity_id: str, *parts: str) -> str:
        return f"{entity_id}::" + "::".join(parts)

    def _get_counter(self, key: str) -> int:
        if self.store:
            try:
                return int(self.store.get_counter(key))
            except Exception:
                pass
        return int(self._mem_counters.get(key, 0))

    def _set_counter(self, key: str, val: int) -> None:
        if self.store:
            try:
                self.store.set_counter(key, val)
                return
            except Exception:
                pass
        self._mem_counters[key] = val

    def _incr(self, key: str, delta: int = 1) -> int:
        v = self._get_counter(key) + delta
        self._set_counter(key, v)
        return v

    def _get_flag_value(self, entity_type: str, entity_id: str, k: str) -> Optional[str]:
        if self.store:
            try:
                got = self.store.get_flag(entity_type, entity_id, k)  # requires Store(entity_type,entity_id,k)
                if got and "v" in got:
                    return str(got["v"])
            except Exception:
                pass
        return self._mem_flags.get((entity_type, entity_id, k))

    def _set_flag_value(self, entity_type: str, entity_id: str, k: str, v: str) -> None:
        if self.store:
            try:
                self.store.set_flag(entity_type, entity_id, k, v)
                return
            except Exception:
                pass
        self._mem_flags[(entity_type, entity_id, k)] = v

    # ---------- Economics ----------
    def _cogs_for_row(self, row: Dict[str, Any]) -> Optional[float]:
        if self.use_creative_cogs:
            r = row.get("cogs_per_purchase")
            if r is not None:
                try:
                    return float(r)
                except Exception:
                    return None
        if self.use_env_cogs:
            return COGS_PER_PURCHASE
        return None

    # ---------- Metrics ----------
    def compute_metrics(self, row: Dict[str, Any]) -> Metrics:
        return _metrics_from_row_compat(
            row,
            cfg=self.metrics_cfg,
            prefer_roas_field=(self.roas_source == "field"),
            cogs_per_purchase=self._cogs_for_row(row),
            smoothing_epsilon=self.eps,
        )

    # ---------- Minimums & stability ----------
    def _meets_minimums(self, stage: str, m: Metrics) -> bool:
        mins = dict(self.global_mins)
        mins.update(((self.cfg.get(stage, {}) or {}).get("minimums", {}) or {}))
        if mins.get("min_impressions") is not None and (m.impressions or 0) < mins["min_impressions"]:
            return False
        if mins.get("min_clicks") is not None and (m.clicks or 0) < mins["min_clicks"]:
            return False
        if mins.get("min_spend") is not None and (m.spend or 0) < mins["min_spend"]:
            return False
        return True

    def _stable(self, entity_id: str, rule_id: str, condition: bool) -> bool:
        need = max(1, self.consecutive_ticks)
        key = self._key(entity_id, "stable", rule_id)
        if condition:
            val = self._incr(key, 1)
            return val >= need
        self._set_counter(key, 0)
        return False

    # ---------- Account-level adaptives ----------
    def _ewma(self, key: str, value: float, alpha: float = 0.2, default: float = 0.0) -> float:
        prev = _f(self._get_flag_value("engine", "GLOBAL", key), default)
        new = alpha * value + (1 - alpha) * prev
        self._set_flag_value("engine", "GLOBAL", key, f"{new:.10f}")
        return new

    def adaptive_bands(self, account_rows: Iterable[Dict[str, Any]]) -> AdaptiveBands:
        m = _aggregate_rows_compat(list(account_rows), cfg=self.metrics_cfg, smoothing_epsilon=self.eps)
        ctr_floor = max(0.003, self._ewma("acct::ctr_floor", m.ctr or 0.0))
        cpa_max = max(10.0, self._ewma("acct::cpa_max", (m.cpa or BREAKEVEN_CPA) * 1.0))
        roas_min = max(0.5, self._ewma("acct::roas_min", m.roas or 1.0))
        return AdaptiveBands(ctr_floor=ctr_floor, cpa_max=cpa_max, roas_min=roas_min)

    # ---------- Creative compliance ----------
    def creative_compliance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        for f in self.cc_require_fields:
            if not str(payload.get(f) or "").strip():
                issues.append(f"Missing required field: {f}")
        blob = " ".join(str(payload.get(k, "")) for k in ("primary_text", "headline", "description")).lower()
        for bad in self.cc_forbid_terms:
            if bad and bad in blob:
                issues.append(f"Contains forbidden term: {bad}")
        for k, max_len in self.cc_max_lengths.items():
            val = str(payload.get(k, "") or "")
            if max_len and len(val) > max_len:
                issues.append(f"{k} exceeds max length {max_len}")
        return {"ok": len(issues) == 0, "issues": issues}

    # ---------- Stage rule evaluation ----------
    def _eval(self, rule: Dict[str, Any], m: Metrics, row: Dict[str, Any]) -> Tuple[bool, str]:
        t = rule["type"]

        if t == "spend_no_purchase":
            ok = (m.spend or 0) >= rule["spend_gte"] and (m.purchases or 0) == 0
            return ok, f"Spend≥€{rule['spend_gte']} & 0 purchases" if ok else ""

        if t == "ctr_below":
            sg = rule.get("spend_gte")
            cond = (sg is None or (m.spend or 0) >= sg) and (m.ctr or 0.0) < rule["ctr_lt"]
            return cond, f"CTR<{rule['ctr_lt']:.2%}" if cond else ""

        if t == "spend_no_atc":
            ok = (m.spend or 0) >= rule["spend_gte"] and (m.add_to_cart or 0) == 0
            return ok, f"Spend≥€{rule['spend_gte']} & 0 ATC" if ok else ""

        if t == "spend_multi_atc_no_purchase":
            ok = (m.spend or 0) >= rule["spend_gte"] and (m.add_to_cart or 0) >= rule["atc_gte"] and (m.purchases or 0) == 0
            return ok, "Multi ATC & 0 purchases" if ok else ""

        if t == "roas_below_after_days":
            ok = (m.roas or 0.0) < rule["roas_lt"]
            return ok, f"ROAS<{rule['roas_lt']}" if ok else ""

        if t == "spend_over_2x_price_no_purchase":
            # Tripwire in account currency (EUR): 2 × (USD product price → EUR)
            threshold = tripwire_threshold_account(PRODUCT_PRICE, multiple=2.0, cfg=self.metrics_cfg)
            ok = (m.spend or 0) >= threshold and (m.purchases or 0) == 0
            return ok, f"Spend≥2× price (€{threshold:.2f}) & 0 purchases" if ok else ""

        if t == "purchase_gte":
            ok = (m.purchases or 0) >= rule["purchases_gte"]
            return ok, f"Purchases≥{rule['purchases_gte']}" if ok else ""

        if t == "cpa_lte":
            ok = (m.cpa is not None) and (m.cpa <= rule["cpa_lte"])
            return ok, f"CPA≤€{rule['cpa_lte']}" if ok else ""

        if t == "ctr_gte":
            ok = (m.ctr or 0.0) >= rule["ctr_gte"]
            return ok, f"CTR≥{rule['ctr_gte']:.2%}" if ok else ""

        if t == "atc_lt":
            ok = (m.add_to_cart or 0) < rule["atc_lt"]
            return ok, f"ATC<{rule['atc_lt']}" if ok else ""

        if t == "spend_no_purchase_with_conditions":
            # Support for performance-based killing with CTR/ATC conditions
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            purchase_ok = (m.purchases or 0) == 0
            
            # Check CTR conditions if specified
            ctr_ok = True
            if "ctr_gte" in rule:
                ctr_ok = ctr_ok and (m.ctr or 0.0) >= rule["ctr_gte"]
            if "ctr_lt" in rule:
                ctr_ok = ctr_ok and (m.ctr or 0.0) < rule["ctr_lt"]
            
            # Check ATC conditions if specified  
            atc_ok = True
            if "atc_gte" in rule:
                atc_ok = atc_ok and (m.add_to_cart or 0) >= rule["atc_gte"]
            if "atc_lt" in rule:
                atc_ok = atc_ok and (m.add_to_cart or 0) < rule["atc_lt"]
            
            ok = spend_ok and purchase_ok and ctr_ok and atc_ok
            return ok, f"Spend≥€{rule['spend_gte']} & 0 purchases & conditions met" if ok else ""

        if t == "learning_acceleration_high_ctr":
            # Accelerate learning for high-CTR ads even without ATC
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            purchase_ok = (m.purchases or 0) == 0
            ctr_ok = (m.ctr or 0.0) >= rule["ctr_gte"]
            ok = spend_ok and purchase_ok and ctr_ok
            return ok, f"High CTR learning acceleration: Spend≥€{rule['spend_gte']} & CTR≥{rule['ctr_gte']:.1%}" if ok else ""

        if t == "learning_acceleration_multi_atc":
            # Massive budget for ads with multiple ATCs
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            purchase_ok = (m.purchases or 0) == 0
            atc_ok = (m.add_to_cart or 0) >= rule["atc_gte"]
            ok = spend_ok and purchase_ok and atc_ok
            return ok, f"Multi-ATC learning: Spend≥€{rule['spend_gte']} & ATC≥{rule['atc_gte']}" if ok else ""

        if t == "zero_performance_quick_kill":
            # Kill zero-CTR ads immediately to save budget
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            purchase_ok = (m.purchases or 0) == 0
            ctr_ok = (m.ctr or 0.0) < rule["ctr_lt"]
            ok = spend_ok and purchase_ok and ctr_ok
            return ok, f"Zero performance kill: Spend≥€{rule['spend_gte']} & CTR<{rule['ctr_lt']:.1%}" if ok else ""

        if t == "cpa_gte":
            ok = (m.cpa is not None) and (m.cpa >= rule["cpa_gte"])
            return ok, f"CPA≥€{rule['cpa_gte']}" if ok else ""

        if t == "roas_gte":
            ok = (m.roas or 0.0) >= rule["roas_gte"]
            return ok, f"ROAS≥{rule['roas_gte']}" if ok else ""

        if t == "poas_gte":
            ok = (m.poas is not None) and (m.poas >= rule["poas_gte"])
            return ok, f"POAS≥{rule['poas_gte']}" if ok else ""

        if t == "poas_lte_over_days":
            ok = (m.poas is not None) and (m.poas <= rule["poas_lte"])
            return ok, f"POAS≤{rule['poas_lte']} (window)" if ok else ""

        if t == "aov_gte":
            ok = (m.aov is not None) and (m.aov >= rule["aov_gte"])
            return ok, f"AOV≥€{rule['aov_gte']}" if ok else ""

        if t == "cpa_gte_consecutive_days":
            ok = (m.cpa is not None) and (m.cpa >= rule["cpa_gte"])
            return ok, f"CPA≥€{rule['cpa_gte']}" if ok else ""

        if t == "roas_lt_over_days":
            ok = (m.roas or 0.0) < rule["roas_lt"]
            return ok, f"ROAS<{rule['roas_lt']} (window)" if ok else ""

        if t == "cpa_over_days":
            ok = (m.cpa is not None) and (m.cpa > rule["cpa_gt"])
            return ok, f"CPA>€{rule['cpa_gt']} (window)" if ok else ""

        if t == "cpa_lte_and_roas_gte_over_days":
            ok = (m.cpa is not None and m.cpa <= rule["cpa_lte"]) and ((m.roas or 0.0) >= rule["roas_gte"])
            return ok, f"CPA≤€{rule['cpa_lte']}&ROAS≥{rule['roas_gte']} (window)" if ok else ""

        if t == "purchases_in_day":
            ok = (m.purchases or 0) >= rule["purchases_gte"] and (m.cpa is not None and m.cpa <= rule["cpa_lte"])
            return ok, f"{int(m.purchases)} purchases & CPA≤€{rule['cpa_lte']}" if ok else ""

        if t == "missing_actions_alert":
            ok = (m.spend or 0) >= rule.get("min_spend_for_alert", 20) and (m.purchases or 0) == 0 and (m.add_to_cart or 0) == 0
            return ok, "Spend present but no actions" if ok else ""

        # NEW RULE TYPES FOR LEARNING PHASE
        if t == "cpm_increase":
            # Kill if CPM increased by specified percentage
            current_cpm = m.cpm or 0.0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            # This would need historical CPM data - simplified for now
            return spend_ok and current_cpm > 200, f"CPM increase detected: {current_cpm:.2f}" if spend_ok and current_cpm > 200 else ""

        if t == "cpm_above":
            # Kill if CPM above threshold
            current_cpm = m.cpm or 0.0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            cpm_ok = current_cpm > rule["cpm_above"]
            return spend_ok and cpm_ok, f"CPM above threshold: {current_cpm:.2f} > {rule['cpm_above']}" if spend_ok and cpm_ok else ""

        if t == "roas_below":
            # Kill if ROAS below threshold
            current_roas = m.roas or 0.0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            roas_ok = current_roas < rule["roas_lt"]
            return spend_ok and roas_ok, f"ROAS below threshold: {current_roas:.3f} < {rule['roas_lt']}" if spend_ok and roas_ok else ""

        if t == "atc_no_purchase":
            # Kill if ATC but no purchase after specified days
            atc_ok = (m.add_to_cart or 0) >= rule["atc_gte"]
            purchase_ok = (m.purchases or 0) == 0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            # Note: days parameter would need to be tracked separately
            return atc_ok and purchase_ok and spend_ok, f"ATC≥{rule['atc_gte']} but no purchase after {rule.get('days', 3)}d" if atc_ok and purchase_ok and spend_ok else ""

        if t == "atc_gte":
            # Promote if ATC meets threshold
            atc_ok = (m.add_to_cart or 0) >= rule["atc_gte"]
            spend_ok = (m.spend or 0) >= rule.get("spend_gte", 0)
            return atc_ok and spend_ok, f"ATC≥{rule['atc_gte']} after €{rule.get('spend_gte', 0)}" if atc_ok and spend_ok else ""

        if t == "cpm_lte":
            # Promote if CPM below threshold
            current_cpm = m.cpm or 0.0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            ctr_ok = (m.ctr or 0.0) >= rule.get("ctr_gte", 0)
            cpm_ok = current_cpm <= rule["cpm_lte"]
            return spend_ok and ctr_ok and cpm_ok, f"CPM≤{rule['cpm_lte']} & CTR≥{rule.get('ctr_gte', 0):.1%}" if spend_ok and ctr_ok and cpm_ok else ""

        if t == "atc_rate_below":
            # Kill if ATC rate below threshold
            atc_rate = (m.add_to_cart or 0) / max(1, m.impressions or 1)
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            rate_ok = atc_rate < rule["atc_rate_lt"]
            return spend_ok and rate_ok, f"ATC rate<{rule['atc_rate_lt']:.1%}" if spend_ok and rate_ok else ""

        if t == "cost_per_atc_above":
            # Kill if cost per ATC above threshold
            cost_per_atc = (m.spend or 0) / max(1, m.add_to_cart or 1)
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            cost_ok = cost_per_atc >= rule["cost_per_atc_gte"]
            return spend_ok and cost_ok, f"Cost per ATC≥€{rule['cost_per_atc_gte']}" if spend_ok and cost_ok else ""

        if t == "atc_no_ic":
            # Kill if ATC but no IC
            atc_ok = (m.add_to_cart or 0) >= rule["atc_gte"]
            ic_ok = (m.initiate_checkout or 0) < rule["ic_lt"]
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            return atc_ok and ic_ok and spend_ok, f"ATC≥{rule['atc_gte']} but IC<{rule['ic_lt']}" if atc_ok and ic_ok and spend_ok else ""

        if t == "max_runtime":
            # Kill after maximum runtime (would need to track ad creation time)
            # This is a placeholder - actual implementation would need ad creation timestamp
            return False, ""

        if t == "mandatory_kill_after_days":
            # Kill after specified days (would need to track ad creation time)
            # This is a placeholder - actual implementation would need ad creation timestamp
            return False, ""

        if t == "cpm_spike":
            # Kill if CPM spikes (would need historical data)
            # This is a placeholder - actual implementation would need historical CPM data
            return False, ""

        if t == "impression_drop":
            # Kill if impressions drop (would need historical data)
            # This is a placeholder - actual implementation would need historical impression data
            return False, ""

        if t == "atc_rate_drop":
            # Kill if ATC rate drops (would need historical data)
            # This is a placeholder - actual implementation would need historical ATC rate data
            return False, ""

        return False, ""

    # ---------- Stage decisions ----------
    def should_kill_testing(self, row: Dict[str, Any], entity_id: str = "ad") -> Tuple[bool, str]:
        m = self.compute_metrics(row)
        if not self._meets_minimums("testing", m):
            return False, ""
        for r in self.cfg.get("testing", {}).get("kill", []):
            ok, reason = self._eval(r, m, row)
            if ok and self._stable(entity_id, f"test_kill:{r['type']}", True):
                return True, reason
            self._stable(entity_id, f"test_kill:{r['type']}", False)
        return False, ""

    def should_advance_from_testing(self, row: Dict[str, Any], entity_id: str = "ad") -> Tuple[bool, str]:
        m = self.compute_metrics(row)
        adv = (self.cfg.get("testing", {}) or {}).get("advance", {}) or {}
        op = (adv.get("operator") or "all").lower()
        rules = adv.get("rules", []) or []
        if not rules:
            return False, ""
        res = [self._eval(r, m, row) for r in rules]
        if (any(ok for ok, _ in res) if op == "any" else all(ok for ok, _ in res)):
            if self._stable(entity_id, f"test_adv:{op}", True):
                reasons = ", ".join([txt for ok, txt in res if ok])
                return True, reasons
            self._stable(entity_id, f"test_adv:{op}", False)
        return False, ""

    def should_kill_validation(self, row: Dict[str, Any], entity_id: str = "ad") -> Tuple[bool, str]:
        m = self.compute_metrics(row)
        if not self._meets_minimums("validation", m):
            return False, ""
        for r in self.cfg.get("validation", {}).get("kill", []):
            ok, reason = self._eval(r, m, row)
            if ok and self._stable(entity_id, f"valid_kill:{r['type']}", True):
                return True, reason
            self._stable(entity_id, f"valid_kill:{r['type']}", False)
        return False, ""

    def _purchase_days(self, row: Dict[str, Any]) -> int:
        return int(row.get("purchase_days", 0) or 0)

    def should_advance_from_validation(self, row: Dict[str, Any], entity_id: str = "ad") -> Tuple[bool, str]:
        m = self.compute_metrics(row)
        adv = (self.cfg.get("validation", {}) or {}).get("advance", {}) or {}
        op = (adv.get("operator") or "all").lower()
        rules = adv.get("rules", []) or []
        strict = (self.cfg.get("validation", {}).get("strictness", {}) or {})
        if not rules:
            return False, ""
        res = [self._eval(r, m, row) for r in rules]
        ok = any(ok for ok, _ in res) if op == "any" else all(ok for ok, _ in res)
        if not ok:
            return False, ""
        need_days = int(strict.get("min_days_with_purchase", 0) or 0)
        if need_days and self._purchase_days(row) < need_days:
            return False, f"Needs purchases on {need_days} days"
        if self._stable(entity_id, f"valid_adv:{op}", True):
            reasons = ", ".join([txt for f, txt in res if f])
            return True, reasons
        self._stable(entity_id, f"valid_adv:{op}", False)
        return False, ""

    def should_kill_scaling(self, row: Dict[str, Any], entity_id: str = "ad") -> Tuple[bool, str]:
        m = self.compute_metrics(row)
        if not self._meets_minimums("scaling", m):
            return False, ""
        consec_key = self._key(entity_id, "scale_bad_days")
        for r in self.cfg.get("scaling", {}).get("kill", []):
            ok, reason = self._eval(r, m, row)
            if r["type"] == "cpa_gte_consecutive_days":
                if m.cpa is not None and m.cpa >= r["cpa_gte"]:
                    days = self._incr(consec_key, 1)
                    if days >= int(r.get("days", 2)):
                        return True, f"{reason} for {days}d"
                else:
                    self._set_counter(consec_key, 0)
            elif ok and self._stable(entity_id, f"scale_kill:{r['type']}", True):
                return True, reason
            self._stable(entity_id, f"scale_kill:{r['type']}", False)
        return False, ""

    # ---------- Scaling actions (legacy YAML keys still supported) ----------
    def scaling_increase_budget_pct(self, row: Dict[str, Any], entity_id: str = "ad") -> int:
        """
        Reads thresholds from scaling.scale_up + scaling.scale_up.hysteresis (legacy keys kept in rules.yaml).
        scaling.py uses scaling.engine.*; this keeps compatibility for any callers still using RuleEngine.
        """
        m = self.compute_metrics(row)
        cfg = (self.cfg.get("scaling", {}).get("scale_up", {}) or {})
        steps = cfg.get("steps", []) or []
        cool_h = int(cfg.get("cooldown_hours", 0) or 0)
        hyst = cfg.get("hysteresis", {}) or {}
        roas_band = float(hyst.get("roas_down_band", 0) or 0)
        cpa_band = float(hyst.get("cpa_up_band", 9e9) or 9e9)

        last_scale_iso = self._get_flag_value("ad", entity_id, "last_scale_ts")
        if last_scale_iso:
            try:
                if (_utcnow() - datetime.fromisoformat(last_scale_iso)) < timedelta(hours=cool_h):
                    return 0
            except Exception:
                pass

        if (m.roas or 0.0) < roas_band or (m.cpa is not None and m.cpa > cpa_band):
            return 0

        for r in steps:
            ok, _ = self._eval(r, m, row)
            if ok:
                pct = int(sorted(r.get("budget_increase_pct", [0]))[0])
                if pct > 0:
                    self._set_flag_value("ad", entity_id, "last_scale_ts", _utcnow().isoformat())
                return pct
        return 0

    def scaling_duplicate_on_fire(self, row: Dict[str, Any], entity_id: str = "ad") -> int:
        m = self.compute_metrics(row)
        dup = (self.cfg.get("scaling", {}).get("duplicate_on_fire", {}) or {})
        cap = int(dup.get("max_duplicates_per_24h", 3) or 3)
        rules = dup.get("rules", []) or []
        used_key = self._key(entity_id, "dups", self._today_key_account())
        used = self._get_counter(used_key)
        for r in rules:
            ok, _ = self._eval(r, m, row)
            if ok:
                want = int(r.get("duplicates", 0) or 0)
                allow = max(0, min(want, cap - used))
                if allow > 0:
                    self._set_counter(used_key, used + allow)
                    return allow
        return 0

    # ---------- Pacing & guardrails ----------
    def pacing_flags(self, spend_today: float, spend_yday: float, spend_48ago: float) -> Dict[str, Any]:
        vcfg = ((self.cfg.get("pacing", {}) or {}).get("spend_velocity", {}) or {})
        g24 = _pct_growth(spend_today, spend_yday)
        g48 = _pct_growth(spend_today + spend_yday, spend_yday + spend_48ago)
        warn24 = g24 >= float(vcfg.get("growth_pct_24h_warn", 150))
        warn48 = g48 >= float(vcfg.get("growth_pct_48h_warn", 250))
        freeze = bool(vcfg.get("freeze_scaling_if_exceeds", True)) and (warn24 or warn48)
        return {"warn_24h": warn24, "warn_48h": warn48, "freeze_scaling": freeze, "g24": g24, "g48": g48}

    def account_guardrails(self, account_cpa: Optional[float], hours_since_last_purchase: Optional[int]) -> Dict[str, Any]:
        acc = ((self.cfg.get("safety_nets", {}) or {}).get("account", {}) or {})
        out = {"pause_scaling": False, "reasons": []}
        pause = acc.get("pause_scaling_if_account_cpa_high", {}) or {}
        if bool(pause.get("enabled", True)) and account_cpa is not None:
            if account_cpa > BREAKEVEN_CPA * float(pause.get("factor_over_breakeven", 1.5)):
                out["pause_scaling"] = True
                out["reasons"].append(f"CPA {account_cpa:.2f} > BE * {pause.get('factor_over_breakeven',1.5)}")
        trip = acc.get("no_purchases_tripwire_hours", None)
        if trip is not None and hours_since_last_purchase is not None and hours_since_last_purchase >= int(trip):
            out["pause_scaling"] = True
            out["reasons"].append(f"No purchases ≥{trip}h")
        return out

    # ---------- Fatigue & reinvest ----------
    def fatigue_flag(self, window_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(window_rows)
        if len(rows) < 2:
            return {"fatigued": False}
        m_old = self.compute_metrics(rows[0])
        m_new = self.compute_metrics(rows[-1])
        ctr_drop = (m_old.ctr or 0) > 0 and ((m_old.ctr - (m_new.ctr or 0)) / (m_old.ctr)) > 0.25
        roas_drop = (m_old.roas or 0) > 0 and ((m_old.roas - (m_new.roas or 0)) / (m_old.roas)) > 0.25
        return {"fatigued": ctr_drop or roas_drop, "ctr_drop": ctr_drop, "roas_drop": roas_drop}

    def reinvest_allocation(self, winners: List[Dict[str, Any]], freed_budget: float, strategy: str = "proportional_to_roas") -> List[Tuple[str, float]]:
        if not winners or freed_budget <= 0:
            return []
        if strategy == "even":
            bump = freed_budget / len(winners)
            return [(w["adset_id"], bump) for w in winners]
        if strategy == "bias_to_new_winners":
            ages = [max(1, int(w.get("days_live", 1))) for w in winners]
            weights = [1.0 / a for a in ages]
            sm = sum(weights) or 1.0
            return [(w["adset_id"], freed_budget * (wgt / sm)) for w, wgt in zip(winners, weights)]
        roas_vals = [max(0.1, _f(w.get("ROAS"), 0.1)) for w in winners]
        probs = _softmax(roas_vals)
        return [(w["adset_id"], freed_budget * p) for w, p in zip(winners, probs)]

    # ---------- Engine accessors (NEW) ----------
    def get_validation_engine_settings(self) -> Dict[str, Any]:
        """
        Convenience accessor mirroring rules.yaml: validation.engine.*
        Helpful if you want to wire env vars for validation.py at runtime
        or inspect configured thresholds from a single place.
        """
        return (self.cfg.get("validation", {}) or {}).get("engine", {}) or {}

    def get_scaling_engine_settings(self) -> Dict[str, Any]:
        """
        Convenience accessor mirroring rules.yaml: scaling.engine.*
        scaling.py reads the same structure directly; this helper is
        provided for tooling/tests that only have RuleEngine around.
        """
        return (self.cfg.get("scaling", {}) or {}).get("engine", {}) or {}

    def get_engine_attr_windows(self, stage: str) -> List[str]:
        """
        Return attribution windows from the stage's engine block if present,
        else fall back to ["7d_click","1d_view"].
        stage: "testing" | "validation" | "scaling"
        """
        stage = (stage or "").lower()
        default = ["7d_click", "1d_view"]
        eng = (self.cfg.get(stage, {}) or {}).get("engine", {}) or {}
        wins = eng.get("attribution_windows")
        if isinstance(wins, (list, tuple)) and wins:
            return list(wins)
        return default

    def account_timezone(self) -> str:
        return self.account_tz_name

    # ---------- Explain / simulate ----------
    def explain(self, stage: str, row: Dict[str, Any]) -> Dict[str, Any]:
        m = self.compute_metrics(row)
        thresholds = self.cfg.get("thresholds", {}) or {}
        return {
            "stage": stage,
            "spend": m.spend,
            "impressions": m.impressions,
            "clicks": m.clicks,
            "purchases": m.purchases,
            "ctr": m.ctr,
            "cpa": m.cpa,
            "roas": m.roas,
            "aov": m.aov,
            "poas": m.poas,
            "thresholds": thresholds,
            "account_tz": self.account_tz_name,
        }

    def simulate_decision(self, stage: str, row: Dict[str, Any]) -> Dict[str, Any]:
        if stage == "TEST":
            k, kr = self.should_kill_testing(row, row.get("ad_id", "ad"))
            a, ar = self.should_advance_from_testing(row, row.get("ad_id", "ad"))
        elif stage == "VALID":
            k, kr = self.should_kill_validation(row, row.get("ad_id", "ad"))
            a, ar = self.should_advance_from_validation(row, row.get("ad_id", "ad"))
        else:
            k, kr = self.should_kill_scaling(row, row.get("ad_id", "ad"))
            inc = self.scaling_increase_budget_pct(row, row.get("ad_id", "ad"))
            dups = self.scaling_duplicate_on_fire(row, row.get("ad_id", "ad"))
            return {"kill": k, "kill_reason": kr, "scale_up_pct": inc, "duplicate": dups, "explain": self.explain(stage, row)}
        return {"kill": k, "kill_reason": kr, "advance": a, "advance_reason": ar, "explain": self.explain(stage, row)}


RuleEngine = AdvancedRuleEngine
