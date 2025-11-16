from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytz

from analytics.metrics import (
    MetricsConfig,
    Metrics,
    metrics_from_row,
    aggregate_rows,
    tripwire_threshold_account,
)

UTC = timezone.utc

PRODUCT_PRICE = float(os.getenv("PRODUCT_PRICE", "50") or 50)
BREAKEVEN_CPA = float(os.getenv("BREAKEVEN_CPA", "27.51") or 27.51)
COGS_ENV = os.getenv("COGS_PER_PURCHASE")
COGS_PER_PURCHASE = float(COGS_ENV) if COGS_ENV not in (None, "") else None


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
        )
    except TypeError:
        try:
            return metrics_from_row(
                row,
                cfg=cfg,
                cogs_per_purchase=cogs_per_purchase,
                smoothing_epsilon=smoothing_epsilon,
            )
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
    def __init__(self, cfg: Dict[str, Any], store: Optional[Any] = None) -> None:
        self.cfg = cfg or {}
        self.store = store
        self._mem_counters: Dict[str, int] = {}
        self._mem_flags: Dict[Tuple[str, str, str], str] = {}

        self.mode = (self.cfg.get("mode", {}).get("current") or "production").lower()

        stab = self.cfg.get("stability", {}) or {}
        self.consecutive_ticks = int(stab.get("consecutive_ticks", 1) or 1)
        self.eps = float(stab.get("smoothing_epsilon", 0.0) or 0.0)

        self.global_mins = self.cfg.get("minimums", {}) or {}
        attr = self.cfg.get("attribution", {}) or {}
        self.roas_source = (attr.get("roas_source") or "computed").lower()

        tz_cfg = self.cfg.get("timezones", {}) or {}
        self.account_tz_name = tz_cfg.get("account_tz") or "Europe/Amsterdam"
        self.audience_tz_name = tz_cfg.get("audience_tz") or "America/Chicago"
        try:
            self.account_tz = pytz.timezone(self.account_tz_name)
        except Exception:
            self.account_tz = pytz.timezone("Europe/Amsterdam")

        self.metrics_cfg = MetricsConfig(
            prefer_roas_field=(self.roas_source == "field"),
            account_currency=(self.cfg.get("economics", {}).get("account_currency") or "EUR"),
            product_currency=(self.cfg.get("economics", {}).get("product_currency") or "EUR"),
        )

        econ = self.cfg.get("economics", {}) or {}
        self.use_env_cogs = bool(econ.get("use_env_cogs", True))
        self.use_creative_cogs = bool(econ.get("use_creative_cogs", False))

        cc = self.cfg.get("creative_compliance") or {}
        self.cc_require_fields: List[str] = list(cc.get("require_fields") or [])
        self.cc_forbid_terms: List[str] = [str(t).lower() for t in (cc.get("forbid_terms") or [])]
        self.cc_max_lengths: Dict[str, int] = {k: int(v) for k, v in (cc.get("max_lengths") or {}).items()}

    def _now_account(self) -> datetime:
        return datetime.now(self.account_tz)

    def _today_key_account(self) -> str:
        return self._now_account().date().isoformat()

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
                got = self.store.get_flag(entity_type, entity_id, k)
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

    def compute_metrics(self, row: Dict[str, Any]) -> Metrics:
        return _metrics_from_row_compat(
            row,
            cfg=self.metrics_cfg,
            prefer_roas_field=(self.roas_source == "field"),
            cogs_per_purchase=self._cogs_for_row(row),
            smoothing_epsilon=self.eps,
        )

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
            threshold = tripwire_threshold_account(PRODUCT_PRICE, multiple=2.0)
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
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            purchase_ok = (m.purchases or 0) == 0
            
            ctr_ok = True
            if "ctr_gte" in rule:
                ctr_ok = ctr_ok and (m.ctr or 0.0) >= rule["ctr_gte"]
            if "ctr_lt" in rule:
                ctr_ok = ctr_ok and (m.ctr or 0.0) < rule["ctr_lt"]
            
            atc_ok = True
            if "atc_gte" in rule:
                atc_ok = atc_ok and (m.add_to_cart or 0) >= rule["atc_gte"]
            if "atc_lt" in rule:
                atc_ok = atc_ok and (m.add_to_cart or 0) < rule["atc_lt"]
            
            ok = spend_ok and purchase_ok and ctr_ok and atc_ok
            return ok, f"Spend≥€{rule['spend_gte']} & 0 purchases & conditions met" if ok else ""

        if t == "learning_acceleration_high_ctr":
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            purchase_ok = (m.purchases or 0) == 0
            ctr_ok = (m.ctr or 0.0) >= rule["ctr_gte"]
            ok = spend_ok and purchase_ok and ctr_ok
            return ok, f"High CTR learning acceleration: Spend≥€{rule['spend_gte']} & CTR≥{rule['ctr_gte']:.1%}" if ok else ""

        if t == "learning_acceleration_multi_atc":
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            purchase_ok = (m.purchases or 0) == 0
            atc_ok = (m.add_to_cart or 0) >= rule["atc_gte"]
            ok = spend_ok and purchase_ok and atc_ok
            return ok, f"Multi-ATC learning: Spend≥€{rule['spend_gte']} & ATC≥{rule['atc_gte']}" if ok else ""

        if t == "zero_performance_quick_kill":
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

        if t == "cpm_increase":
            ad_id = row.get("ad_id")
            if not ad_id:
                return False, ""
            
            store = getattr(self, 'store', None)
            if store:
                historical_cpm = store.get_historical_data(ad_id, "cpm", since_days=2)
                if len(historical_cpm) >= 1:
                    current_cpm = m.cpm or 0.0
                    spend_ok = (m.spend or 0) >= rule["spend_gte"]
                    avg_cpm = sum(h["metric_value"] for h in historical_cpm) / len(historical_cpm)
                    increase_threshold = rule.get("cpm_increase_pct", 80) / 100.0
                    
                    if spend_ok and current_cpm > avg_cpm * (1 + increase_threshold):
                        return True, f"CPM increase: {current_cpm:.2f} vs avg {avg_cpm:.2f} (+{increase_threshold*100:.0f}%)"
            return False, ""

        if t == "cpm_above":
            current_cpm = m.cpm or 0.0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            cpm_ok = current_cpm > rule["cpm_above"]
            return spend_ok and cpm_ok, f"CPM above threshold: {current_cpm:.2f} > {rule['cpm_above']}" if spend_ok and cpm_ok else ""

        if t == "roas_below":
            current_roas = m.roas or 0.0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            roas_ok = current_roas < rule["roas_lt"]
            return spend_ok and roas_ok, f"ROAS below threshold: {current_roas:.3f} < {rule['roas_lt']}" if spend_ok and roas_ok else ""

        if t == "atc_no_purchase":
            atc_ok = (m.add_to_cart or 0) >= rule["atc_gte"]
            purchase_ok = (m.purchases or 0) == 0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            return atc_ok and purchase_ok and spend_ok, f"ATC≥{rule['atc_gte']} but no purchase after {rule.get('days', 3)}d" if atc_ok and purchase_ok and spend_ok else ""

        if t == "atc_gte":
            atc_ok = (m.add_to_cart or 0) >= rule["atc_gte"]
            spend_ok = (m.spend or 0) >= rule.get("spend_gte", 0)
            return atc_ok and spend_ok, f"ATC≥{rule['atc_gte']} after €{rule.get('spend_gte', 0)}" if atc_ok and spend_ok else ""

        if t == "cpm_lte":
            current_cpm = m.cpm or 0.0
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            ctr_ok = (m.ctr or 0.0) >= rule.get("ctr_gte", 0)
            cpm_ok = current_cpm <= rule["cpm_lte"]
            return spend_ok and ctr_ok and cpm_ok, f"CPM≤{rule['cpm_lte']} & CTR≥{rule.get('ctr_gte', 0):.1%}" if spend_ok and ctr_ok and cpm_ok else ""

        if t == "atc_rate_below":
            atc_rate = (m.add_to_cart or 0) / max(1, m.impressions or 1)
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            rate_ok = atc_rate < rule["atc_rate_lt"]
            return spend_ok and rate_ok, f"ATC rate<{rule['atc_rate_lt']:.1%}" if spend_ok and rate_ok else ""

        if t == "cost_per_atc_above":
            cost_per_atc = (m.spend or 0) / max(1, m.add_to_cart or 1)
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            cost_ok = cost_per_atc >= rule["cost_per_atc_gte"]
            return spend_ok and cost_ok, f"Cost per ATC≥€{rule['cost_per_atc_gte']}" if spend_ok and cost_ok else ""

        if t == "atc_no_ic":
            atc_ok = (m.add_to_cart or 0) >= rule["atc_gte"]
            ic_ok = (m.initiate_checkout or 0) < rule["ic_lt"]
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            return atc_ok and ic_ok and spend_ok, f"ATC≥{rule['atc_gte']} but IC<{rule['ic_lt']}" if atc_ok and ic_ok and spend_ok else ""

        if t == "max_runtime":
            ad_id = row.get("ad_id")
            if not ad_id:
                return False, ""
            
            store = getattr(self, 'store', None)
            if store:
                ad_age_days = store.get_ad_age_days(ad_id)
                if ad_age_days is not None and ad_age_days >= rule.get("days", 7):
                    return True, f"Ad runtime≥{rule.get('days', 7)}d (age: {ad_age_days:.1f}d)"
            return False, ""

        if t == "mandatory_kill_after_days":
            ad_id = row.get("ad_id")
            if not ad_id:
                return False, ""
            
            store = getattr(self, 'store', None)
            if store:
                ad_age_days = store.get_ad_age_days(ad_id)
                if ad_age_days is not None and ad_age_days >= rule.get("days", 7):
                    return True, f"Mandatory kill after {rule.get('days', 7)}d (age: {ad_age_days:.1f}d)"
            return False, ""

        if t == "ctr_spend_floor":
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            ctr_ok = (m.ctr or 0.0) < rule["ctr_lt"]
            return spend_ok and ctr_ok, f"Spend≥€{rule['spend_gte']} & CTR<{rule['ctr_lt']:.2%}" if spend_ok and ctr_ok else ""

        if t == "cpc_spend_combo":
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            cpc_ok = (m.cpc or 0.0) > rule["cpc_gt"]
            return spend_ok and cpc_ok, f"Spend≥€{rule['spend_gte']} & CPC>€{rule['cpc_gt']}" if spend_ok and cpc_ok else ""

        if t == "cpm_ctr_combo":
            cpm_ok = (m.cpm or 0.0) > rule["cpm_gt"]
            ctr_ok = (m.ctr or 0.0) < rule["ctr_lt"]
            return cpm_ok and ctr_ok, f"CPM>€{rule['cpm_gt']} & CTR<{rule['ctr_lt']:.2%}" if cpm_ok and ctr_ok else ""

        if t == "atc_efficiency_fail":
            spend_ok = (m.spend or 0) >= rule["spend_gte"]
            atc_ok = (m.add_to_cart or 0) < rule["atc_lt"]
            return spend_ok and atc_ok, f"Spend≥€{rule['spend_gte']} & ATC<{rule['atc_lt']}" if spend_ok and atc_ok else ""

        if t == "cpm_spike":
            ad_id = row.get("ad_id")
            if not ad_id:
                return False, ""
            
            store = getattr(self, 'store', None)
            if store:
                historical_cpm = store.get_historical_data(ad_id, "cpm", since_days=3)
                if len(historical_cpm) >= 2:
                    current_cpm = m.cpm or 0.0
                    avg_cpm = sum(h["metric_value"] for h in historical_cpm[1:]) / len(historical_cpm[1:])
                    spike_threshold = rule.get("cpm_spike_pct", 100) / 100.0
                    
                    if current_cpm > avg_cpm * (1 + spike_threshold):
                        return True, f"CPM spike: {current_cpm:.2f} vs avg {avg_cpm:.2f} (+{spike_threshold*100:.0f}%)"
            return False, ""

        if t == "impression_drop":
            ad_id = row.get("ad_id")
            if not ad_id:
                return False, ""
            
            store = getattr(self, 'store', None)
            if store:
                historical_impressions = store.get_historical_data(ad_id, "impressions", since_days=3)
                if len(historical_impressions) >= 2:
                    current_impressions = m.impressions or 0
                    avg_impressions = sum(h["metric_value"] for h in historical_impressions[1:]) / len(historical_impressions[1:])
                    drop_threshold = rule.get("impression_drop_pct", 30) / 100.0
                    
                    if current_impressions < avg_impressions * (1 - drop_threshold):
                        return True, f"Impression drop: {current_impressions} vs avg {avg_impressions:.0f} (-{drop_threshold*100:.0f}%)"
            return False, ""

        if t == "atc_rate_drop":
            ad_id = row.get("ad_id")
            if not ad_id:
                return False, ""
            
            store = getattr(self, 'store', None)
            if store:
                historical_atc_rate = store.get_historical_data(ad_id, "atc_rate", since_days=3)
                if len(historical_atc_rate) >= 2:
                    current_atc_rate = (m.add_to_cart or 0) / max(1, m.impressions or 1)
                    avg_atc_rate = sum(h["metric_value"] for h in historical_atc_rate[1:]) / len(historical_atc_rate[1:])
                    drop_threshold = rule.get("atc_rate_drop_pct", 25) / 100.0
                    
                    if current_atc_rate < avg_atc_rate * (1 - drop_threshold):
                        return True, f"ATC rate drop: {current_atc_rate:.3f} vs avg {avg_atc_rate:.3f} (-{drop_threshold*100:.0f}%)"
            return False, ""

        return False, ""





RuleEngine = AdvancedRuleEngine
