from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import math
import os

@dataclass(frozen=True)
class MetricsConfig:
    action_aliases: Mapping[str, Tuple[str, ...]] = None
    value_aliases: Mapping[str, Tuple[str, ...]] = None
    window_keys: Tuple[str, ...] = ("actions", "conversions")
    value_keys: Tuple[str, ...] = ("action_values",)
    purchase_roas_index: int = 0
    smoothing_epsilon: float = 0.0
    prefer_roas_field: bool = False
    beta_prior_ctr: Tuple[float, float] = (1.0, 100.0)
    beta_prior_cvr: Tuple[float, float] = (1.0, 50.0)
    beta_prior_atc_rate: Tuple[float, float] = (1.0, 50.0)
    ci_z: float = 1.96
    account_currency: str = (os.getenv("ACCOUNT_CURRENCY") or "EUR").upper()
    product_currency: str = (os.getenv("PRODUCT_CURRENCY") or "EUR").upper()

    def __post_init__(self) -> None:
        if self.action_aliases is None:
            object.__setattr__(self, "action_aliases", {
                "purchase": (
                    "purchase",
                    "offsite_conversion.purchase",
                    "onsite_conversion.purchase",
                    "omni_purchase",
                ),
                "add_to_cart": (
                    "add_to_cart",
                    "offsite_conversion.add_to_cart",
                    "onsite_conversion.add_to_cart",
                    "omni_add_to_cart",
                ),
                "link_click": ("link_click", "omni_link_click"),
                "view_content": ("view_content", "omni_view_content"),
                "lead": ("lead", "omni_lead"),
            })
        if self.value_aliases is None:
            object.__setattr__(self, "value_aliases", {
                "purchase": (
                    "purchase",
                    "offsite_conversion.purchase",
                    "onsite_conversion.purchase",
                    "omni_purchase",
                )
            })
        for req in ("purchase", "add_to_cart"):
            if req not in self.action_aliases:
                raise ValueError(f"Missing action_aliases key: {req!r}")
        if "purchase" not in self.value_aliases:
            raise ValueError("Missing value_aliases key: 'purchase'")
        if self.ci_z <= 0:
            raise ValueError("ci_z must be > 0")
        if self.purchase_roas_index < 0:
            raise ValueError("purchase_roas_index must be >= 0")


Number = Union[int, float]


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _scan_actions(row: Dict[str, Any], candidates: Sequence[str], sources: Sequence[str]) -> float:
    total = 0.0
    for src in sources:
        acts = row.get(src) or []
        if not isinstance(acts, list):
            continue
        for a in acts:
            if not isinstance(a, dict):
                continue
            at = a.get("action_type")
            if at in candidates:
                total += _to_float(a.get("value", 0.0))
    return total


def _scan_values(row: Dict[str, Any], candidates: Sequence[str], sources: Sequence[str]) -> float:
    total = 0.0
    for src in sources:
        vals = row.get(src) or []
        if not isinstance(vals, list):
            continue
        for v in vals:
            if not isinstance(v, dict):
                continue
            at = v.get("action_type")
            if at in candidates:
                total += _to_float(v.get("value", 0.0))
    return total


def _safe_div(n: Optional[float], d: Optional[float], default: Optional[float] = None) -> Optional[float]:
    if n is None or d in (None, 0):
        return default
    return n / d


def _beta_smooth(successes: float, trials: float, a: float, b: float) -> float:
    return (successes + a) / (trials + a + b) if trials >= 0 else 0.0


def _wilson_ci(successes: float, trials: float, z: float) -> Tuple[Optional[float], Optional[float]]:
    if trials <= 0:
        return (None, None)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    sqrt_arg = (p * (1 - p) + z * z / (4 * trials)) / trials
    adj = z * math.sqrt(max(0.0, sqrt_arg))
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


def tripwire_threshold_account(product_price: float, multiple: float = 2.0, cfg: Optional[MetricsConfig] = None) -> float:
    return product_price * multiple


@dataclass
class Metrics:
    spend: float = 0.0
    impressions: float = 0.0
    clicks: float = 0.0
    unique_clicks: Optional[float] = None
    reach: Optional[float] = None
    frequency: Optional[float] = None
    purchases: float = 0.0
    add_to_cart: float = 0.0
    initiate_checkout: float = 0.0
    revenue: float = 0.0
    ctr: Optional[float] = None
    ctr_wilson_lo: Optional[float] = None
    ctr_wilson_hi: Optional[float] = None
    ctr_smoothed: Optional[float] = None
    unique_ctr: Optional[float] = None
    cpc: Optional[float] = None
    cpm: Optional[float] = None
    cvr: Optional[float] = None
    cvr_wilson_lo: Optional[float] = None
    cvr_wilson_hi: Optional[float] = None
    cvr_smoothed: Optional[float] = None
    atc_rate: Optional[float] = None
    atc_rate_wilson_lo: Optional[float] = None
    atc_rate_wilson_hi: Optional[float] = None
    atc_rate_smoothed: Optional[float] = None
    aov: Optional[float] = None
    cpa: Optional[float] = None
    roas: Optional[float] = None
    profit: Optional[float] = None
    poas: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Diagnostics:
    used_action_aliases: Mapping[str, str]
    used_value_aliases: Mapping[str, str]
    missing_required: Tuple[str, ...] = ()
    anomalies: Tuple[str, ...] = ()


def metrics_from_row(
    row: Dict[str, Any],
    cfg: Optional[MetricsConfig] = None,
    *,
    cogs_per_purchase: Optional[float] = None,
    smoothing_epsilon: Optional[float] = None,
    compute_diagnostics: bool = False,
) -> Union[Metrics, Tuple[Metrics, Diagnostics]]:
    cfg = cfg or MetricsConfig()

    spend = _to_float(row.get("spend"))
    imps = _to_float(row.get("impressions"))
    clicks = _to_float(row.get("clicks"))
    uniq_clicks = _to_float(row.get("unique_clicks")) if row.get("unique_clicks") is not None else None
    reach = _to_float(row.get("reach")) if row.get("reach") is not None else None
    frequency = _safe_div(imps, reach, None) if (reach and reach > 0) else None

    used_act: Dict[str, str] = {}
    used_val: Dict[str, str] = {}

    # Check if purchases is already a direct field (from _build_ad_metrics)
    # Prefer direct fields over scanning actions array for performance and accuracy
    if "purchases" in row or "purchase" in row:
        purchases = _to_float(row.get("purchases")) or _to_float(row.get("purchase"))
    else:
        # Fall back to scanning actions array if direct field not available
        purchases = _scan_actions(row, cfg.action_aliases["purchase"], cfg.window_keys)
    if purchases > 0:
        for c in cfg.action_aliases["purchase"]:
            if _scan_actions(row, (c,), cfg.window_keys) > 0:
                used_act.setdefault("purchase", c)
                break

    # Check if add_to_cart is already a direct field (from _build_ad_metrics)
    # This is critical because _build_ad_metrics already extracts add_to_cart from actions
    # and the rule engine needs to see the correct ATC value to avoid false positives
    if "add_to_cart" in row or "atc" in row:
        atc = _to_float(row.get("add_to_cart")) or _to_float(row.get("atc"))
    else:
        # Fall back to scanning actions array if direct field not available
        atc = _scan_actions(row, cfg.action_aliases["add_to_cart"], cfg.window_keys)
    if atc > 0:
        for c in cfg.action_aliases["add_to_cart"]:
            if _scan_actions(row, (c,), cfg.window_keys) > 0:
                used_act.setdefault("add_to_cart", c)
                break

    revenue = _scan_values(row, cfg.value_aliases["purchase"], cfg.value_keys)
    if revenue > 0:
        for c in cfg.value_aliases["purchase"]:
            if _scan_values(row, (c,), cfg.value_keys) > 0:
                used_val.setdefault("purchase", c)
                break

    ctr = _safe_div(clicks, imps, None) if clicks >= 0 else None
    ctr_lo, ctr_hi = _wilson_ci(clicks, imps, cfg.ci_z)
    ctr_sm = _beta_smooth(clicks, imps, *cfg.beta_prior_ctr)

    unique_ctr = _safe_div((uniq_clicks or 0.0), (reach or 0.0), None) if (uniq_clicks is not None and reach) else None
    cpc = _safe_div(spend, clicks, None) if clicks > 0 else None
    cpm = _safe_div(spend * 1000.0, imps, None)
    cvr = _safe_div(purchases, clicks, None) if clicks > 0 else None
    cvr_lo, cvr_hi = _wilson_ci(purchases, clicks, cfg.ci_z) if clicks > 0 else (None, None)
    cvr_sm = _beta_smooth(purchases, clicks, *cfg.beta_prior_cvr)

    atc_rate = _safe_div(atc, clicks, None) if clicks > 0 else None
    atc_lo, atc_hi = _wilson_ci(atc, clicks, cfg.ci_z) if clicks > 0 else (None, None)
    atc_sm = _beta_smooth(atc, clicks, *cfg.beta_prior_atc_rate)

    aov = _safe_div(revenue, purchases, None) if purchases > 0 else None
    cpa = _safe_div(spend, purchases, None) if purchases > 0 else None

    roas = _safe_div(revenue, spend, None) if spend > 0 else None
    if cfg.prefer_roas_field:
        proas = row.get("purchase_roas")
        if isinstance(proas, list) and proas:
            idx = max(0, min(cfg.purchase_roas_index, len(proas) - 1))
            cand = proas[idx]
            if isinstance(cand, dict):
                roas_field = _to_float(cand.get("value"))
            else:
                roas_field = _to_float(cand)
            if roas_field > 0:
                roas = roas_field

    profit = poas = None
    if cogs_per_purchase is not None:
        profit = revenue - (cogs_per_purchase * purchases) - spend
        poas = _safe_div(profit, spend, None) if spend > 0 else None

    m = Metrics(
        spend=spend,
        impressions=imps,
        clicks=clicks,
        unique_clicks=uniq_clicks,
        reach=reach,
        frequency=frequency,
        purchases=purchases,
        add_to_cart=atc,
        revenue=revenue,
        ctr=ctr,
        ctr_wilson_lo=ctr_lo,
        ctr_wilson_hi=ctr_hi,
        ctr_smoothed=ctr_sm,
        unique_ctr=unique_ctr,
        cpc=cpc,
        cpm=cpm,
        cvr=cvr,
        cvr_wilson_lo=cvr_lo,
        cvr_wilson_hi=cvr_hi,
        cvr_smoothed=cvr_sm,
        atc_rate=atc_rate,
        atc_rate_wilson_lo=atc_lo,
        atc_rate_wilson_hi=atc_hi,
        atc_rate_smoothed=atc_sm,
        aov=aov,
        cpa=cpa,
        roas=roas,
        profit=profit,
        poas=poas,
    )

    if not compute_diagnostics:
        return m

    missing: List[str] = []
    if spend < 0 or imps < 0 or clicks < 0:
        missing.append("negative_primitives")
    anomalies: List[str] = []
    if spend > 0 and revenue == 0 and roas == 0:
        anomalies.append("spend_without_revenue")
    if imps == 0 and clicks > 0:
        anomalies.append("clicks_without_impressions")

    d = Diagnostics(
        used_action_aliases=used_act,
        used_value_aliases=used_val,
        missing_required=tuple(missing),
        anomalies=tuple(anomalies),
    )
    return m, d


def aggregate_rows(
    rows: Iterable[Dict[str, Any]],
    cfg: Optional[MetricsConfig] = None,
    *,
    cogs_per_purchase: Optional[float] = None,
    smoothing_epsilon: Optional[float] = None,
) -> Metrics:
    cfg = cfg or MetricsConfig()
    eps = cfg.smoothing_epsilon if smoothing_epsilon is None else smoothing_epsilon

    spend = imps = clicks = purchases = atc = revenue = 0.0
    uniq_clicks_sum = reach_sum = 0.0
    ctr_clicks_sum = ctr_imps_sum = 0.0
    cvr_purch_sum = cvr_clicks_sum = 0.0
    atc_sum = atc_clicks_sum = 0.0

    for r in rows:
        m = metrics_from_row(r, cfg, cogs_per_purchase=None, smoothing_epsilon=eps)
        spend += m.spend
        imps += m.impressions
        clicks += m.clicks
        purchases += m.purchases
        atc += m.add_to_cart
        revenue += m.revenue
        uniq_clicks_sum += (m.unique_clicks or 0.0)
        reach_sum += (m.reach or 0.0)
        ctr_clicks_sum += m.clicks
        ctr_imps_sum += m.impressions
        cvr_purch_sum += m.purchases
        cvr_clicks_sum += m.clicks
        atc_sum += m.add_to_cart
        atc_clicks_sum += m.clicks

    ctr = _safe_div(clicks, imps, None) if clicks >= 0 else None
    ctr_lo, ctr_hi = _wilson_ci(ctr_clicks_sum, ctr_imps_sum, cfg.ci_z)
    ctr_sm = _beta_smooth(ctr_clicks_sum, ctr_imps_sum, *cfg.beta_prior_ctr)

    unique_ctr = _safe_div(uniq_clicks_sum, reach_sum, None) if reach_sum > 0 else None
    cpc = _safe_div(spend, clicks, None) if clicks > 0 else None
    cpm = _safe_div(spend * 1000.0, imps, None)
    cvr = _safe_div(cvr_purch_sum, cvr_clicks_sum if cvr_clicks_sum > 0 else None, None)
    cvr_lo, cvr_hi = _wilson_ci(cvr_purch_sum, cvr_clicks_sum, cfg.ci_z) if cvr_clicks_sum > 0 else (None, None)
    cvr_sm = _beta_smooth(cvr_purch_sum, cvr_clicks_sum, *cfg.beta_prior_cvr)

    atc_rate = _safe_div(atc_sum, atc_clicks_sum if atc_clicks_sum > 0 else None, None)
    atc_lo, atc_hi = _wilson_ci(atc_sum, atc_clicks_sum, cfg.ci_z) if atc_clicks_sum > 0 else (None, None)
    atc_sm = _beta_smooth(atc_sum, atc_clicks_sum, *cfg.beta_prior_atc_rate)

    aov = _safe_div(revenue, purchases, None) if purchases > 0 else None
    cpa = _safe_div(spend, purchases, None) if purchases > 0 else None
    roas = _safe_div(revenue, spend, None) if spend > 0 else None

    profit = poas = None
    if cogs_per_purchase is not None:
        profit = revenue - (cogs_per_purchase * purchases) - spend
        poas = _safe_div(profit, spend, None) if spend > 0 else None

    return Metrics(
        spend=spend,
        impressions=imps,
        clicks=clicks,
        unique_clicks=uniq_clicks_sum or None,
        reach=reach_sum or None,
        frequency=_safe_div(imps, reach_sum, None) if reach_sum > 0 else None,
        purchases=purchases,
        add_to_cart=atc,
        revenue=revenue,
        ctr=ctr,
        ctr_wilson_lo=ctr_lo,
        ctr_wilson_hi=ctr_hi,
        ctr_smoothed=ctr_sm,
        unique_ctr=unique_ctr,
        cpc=cpc,
        cpm=cpm,
        cvr=cvr,
        cvr_wilson_lo=cvr_lo,
        cvr_wilson_hi=cvr_hi,
        cvr_smoothed=cvr_sm,
        atc_rate=atc_rate,
        atc_rate_wilson_lo=atc_lo,
        atc_rate_wilson_hi=atc_hi,
        atc_rate_smoothed=atc_sm,
        aov=aov,
        cpa=cpa,
        roas=roas,
        profit=profit,
        poas=poas,
    )


__all__ = [
    "MetricsConfig",
    "Metrics",
    "Diagnostics",
    "metrics_from_row",
    "aggregate_rows",
    "tripwire_threshold_account",
]
