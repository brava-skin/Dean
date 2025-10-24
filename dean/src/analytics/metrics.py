from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import math
import os

# -------------------- Configuration --------------------

@dataclass(frozen=True)
class MetricsConfig:
    action_aliases: Mapping[str, Tuple[str, ...]] = None  # type: ignore[assignment]
    value_aliases: Mapping[str, Tuple[str, ...]] = None  # type: ignore[assignment]
    window_keys: Tuple[str, ...] = ("actions", "conversions")
    value_keys: Tuple[str, ...] = ("action_values",)
    purchase_roas_index: int = 0
    smoothing_epsilon: float = 0.0
    prefer_roas_field: bool = False
    beta_prior_ctr: Tuple[float, float] = (1.0, 100.0)
    beta_prior_cvr: Tuple[float, float] = (1.0, 50.0)
    beta_prior_atc_rate: Tuple[float, float] = (1.0, 50.0)
    ci_z: float = 1.96

    # NEW: currency defaults to support your Amsterdam / EUR account
    account_currency: str = (os.getenv("ACCOUNT_CURRENCY") or "EUR").upper()
    product_currency: str = (os.getenv("PRODUCT_CURRENCY") or "USD").upper()
    # USD→EUR rate overrideable via env (falls back to 0.92 if not set)
    usd_eur_rate: float = float(os.getenv("USD_EUR_RATE") or os.getenv("EXCHANGE_RATE_USD_EUR") or 0.92)

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


# -------------------- Types & math helpers --------------------

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


def _sum_field_in_list_of_dicts(items: Any, key: str = "value") -> float:
    if not isinstance(items, list):
        return 0.0
    total = 0.0
    for it in items:
        if isinstance(it, dict):
            total += _to_float(it.get(key))
    return total


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


def _safe_div(n: float, d: float, default: Optional[float] = None) -> Optional[float]:
    if d == 0:
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
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


# -------------------- Currency helpers (NEW) --------------------

def convert(amount: float, from_ccy: str, to_ccy: str, cfg: Optional[MetricsConfig] = None) -> float:
    """
    Simple USD<->EUR converter using cfg.usd_eur_rate (override via env: USD_EUR_RATE or EXCHANGE_RATE_USD_EUR).
    """
    cfg = cfg or MetricsConfig()
    f = (from_ccy or "").upper()
    t = (to_ccy or "").upper()
    if f == t:
        return float(amount)
    if f == "USD" and t == "EUR":
        return float(amount) * float(cfg.usd_eur_rate)
    if f == "EUR" and t == "USD":
        rate = 1.0 / float(cfg.usd_eur_rate) if cfg.usd_eur_rate else 0.0
        return float(amount) * rate
    raise ValueError(f"Unsupported conversion {f}->{t} (module handles USD<->EUR only).")


def usd_to_account(amount_usd: float, cfg: Optional[MetricsConfig] = None) -> float:
    """
    Convert a USD amount to the account currency defined in cfg (EUR by default for your setup).
    """
    cfg = cfg or MetricsConfig()
    if cfg.account_currency == "EUR":
        return convert(amount_usd, "USD", "EUR", cfg)
    if cfg.account_currency == "USD":
        return float(amount_usd)
    raise ValueError(f"Unsupported account currency {cfg.account_currency} (expected EUR or USD).")


def tripwire_threshold_account(product_price_usd: float, multiple: float = 2.0, cfg: Optional[MetricsConfig] = None) -> float:
    """
    For rules like: 'Instant tripwire: spend ≥ 2× product price (USD→EUR) & 0 purchases'
    Returns threshold in account currency (EUR for your account).
    """
    cfg = cfg or MetricsConfig()
    return usd_to_account(product_price_usd * multiple, cfg)


# -------------------- Data models --------------------

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
    three_sec_views: Optional[float] = None

    ctr: float = 0.0
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
    roas: float = 0.0
    thumbstop_rate: Optional[float] = None

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


# -------------------- Row → Metrics --------------------

def metrics_from_row(
    row: Dict[str, Any],
    cfg: Optional[MetricsConfig] = None,
    *,
    cogs_per_purchase: Optional[float] = None,
    smoothing_epsilon: Optional[float] = None,
    compute_diagnostics: bool = False,
) -> Union[Metrics, Tuple[Metrics, Diagnostics]]:
    cfg = cfg or MetricsConfig()
    eps = cfg.smoothing_epsilon if smoothing_epsilon is None else smoothing_epsilon

    spend = _to_float(row.get("spend"))
    imps = _to_float(row.get("impressions"))
    clicks = _to_float(row.get("clicks"))
    uniq_clicks = _to_float(row.get("unique_clicks")) if row.get("unique_clicks") is not None else None
    reach = _to_float(row.get("reach")) if row.get("reach") is not None else None
    frequency = _to_float(row.get("frequency")) if row.get("frequency") is not None else (
        _safe_div(imps, reach) if (reach and reach > 0) else None
    )

    used_act: Dict[str, str] = {}
    used_val: Dict[str, str] = {}

    purchases = _scan_actions(row, cfg.action_aliases["purchase"], cfg.window_keys)
    if purchases > 0:
        for c in cfg.action_aliases["purchase"]:
            if _scan_actions(row, (c,), cfg.window_keys) > 0:
                used_act.setdefault("purchase", c)
                break

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

    three_sec_views: Optional[float] = None
    if isinstance(row.get("video_3_sec_views"), (int, float, str)):
        v = _to_float(row.get("video_3_sec_views"))
        three_sec_views = v if v > 0 else None
    elif isinstance(row.get("video_play_actions"), list):
        s = _sum_field_in_list_of_dicts(row["video_play_actions"])
        three_sec_views = s if s > 0 else None

    denom_imps = imps if imps > 0 else (imps + (eps or 0.0))
    denom_clicks = clicks if clicks > 0 else (clicks + (eps or 0.0))

    ctr = (clicks / denom_imps) if denom_imps > 0 else 0.0
    ctr_lo, ctr_hi = _wilson_ci(clicks, imps, cfg.ci_z)
    ctr_sm = _beta_smooth(clicks, imps, *cfg.beta_prior_ctr)

    unique_ctr = _safe_div((uniq_clicks or 0.0), (reach or 0.0), None) if (uniq_clicks is not None and reach) else None
    cpc = _safe_div(spend, denom_clicks, None)
    cpm = _safe_div(spend * 1000.0, denom_imps, None)
    cvr = _safe_div(purchases, denom_clicks, None)
    cvr_lo, cvr_hi = _wilson_ci(purchases, clicks, cfg.ci_z) if clicks > 0 else (None, None)
    cvr_sm = _beta_smooth(purchases, clicks, *cfg.beta_prior_cvr)

    atc_rate = _safe_div(atc, denom_clicks, None)
    atc_lo, atc_hi = _wilson_ci(atc, clicks, cfg.ci_z) if clicks > 0 else (None, None)
    atc_sm = _beta_smooth(atc, clicks, *cfg.beta_prior_atc_rate)

    aov = _safe_div(revenue, purchases, None) if purchases > 0 else None
    cpa = _safe_div(spend, purchases, None) if purchases > 0 else None

    roas = _safe_div(revenue, spend, 0.0) if spend > 0 else 0.0
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

    thumbstop_rate = _safe_div(three_sec_views or 0.0, denom_imps, None) if three_sec_views is not None else None

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
        three_sec_views=three_sec_views,
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
        roas=roas or 0.0,
        thumbstop_rate=thumbstop_rate,
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


# -------------------- Aggregations --------------------

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
    uniq_clicks_sum = reach_sum = three_sec_sum = 0.0
    ctr_clicks_sum = ctr_imps_sum = 0.0
    cvr_purch_sum = cvr_clicks_sum = 0.0
    atc_sum = atc_clicks_sum = 0.0

    for r in rows:
        m = metrics_from_row(r, cfg, cogs_per_purchase=None, smoothing_epsilon=eps)  # type: ignore[assignment]
        spend += m.spend
        imps += m.impressions
        clicks += m.clicks
        purchases += m.purchases
        atc += m.add_to_cart
        revenue += m.revenue
        uniq_clicks_sum += (m.unique_clicks or 0.0)
        reach_sum += (m.reach or 0.0)
        three_sec_sum += (m.three_sec_views or 0.0)
        ctr_clicks_sum += m.clicks
        ctr_imps_sum += m.impressions
        cvr_purch_sum += m.purchases
        cvr_clicks_sum += m.clicks
        atc_sum += m.add_to_cart
        atc_clicks_sum += m.clicks

    denom_imps = imps if imps > 0 else (imps + (eps or 0.0))
    denom_clicks = clicks if clicks > 0 else (clicks + (eps or 0.0))

    ctr = (clicks / denom_imps) if denom_imps > 0 else 0.0
    ctr_lo, ctr_hi = _wilson_ci(ctr_clicks_sum, ctr_imps_sum, cfg.ci_z)
    ctr_sm = _beta_smooth(ctr_clicks_sum, ctr_imps_sum, *cfg.beta_prior_ctr)

    unique_ctr = _safe_div(uniq_clicks_sum, reach_sum, None) if reach_sum > 0 else None
    cpc = _safe_div(spend, denom_clicks, None)
    cpm = _safe_div(spend * 1000.0, denom_imps, None)
    cvr = _safe_div(cvr_purch_sum, cvr_clicks_sum + (eps or 0.0), None) if (cvr_clicks_sum > 0 or eps) else None
    cvr_lo, cvr_hi = _wilson_ci(cvr_purch_sum, cvr_clicks_sum, cfg.ci_z) if cvr_clicks_sum > 0 else (None, None)
    cvr_sm = _beta_smooth(cvr_purch_sum, cvr_clicks_sum, *cfg.beta_prior_cvr)

    atc_rate = _safe_div(atc_sum, atc_clicks_sum + (eps or 0.0), None) if (atc_clicks_sum > 0 or eps) else None
    atc_lo, atc_hi = _wilson_ci(atc_sum, atc_clicks_sum, cfg.ci_z) if atc_clicks_sum > 0 else (None, None)
    atc_sm = _beta_smooth(atc_sum, atc_clicks_sum, *cfg.beta_prior_atc_rate)

    aov = _safe_div(revenue, purchases, None) if purchases > 0 else None
    cpa = _safe_div(spend, purchases, None) if purchases > 0 else None
    roas = _safe_div(revenue, spend, 0.0) if spend > 0 else 0.0
    thumb = _safe_div(three_sec_sum, denom_imps, None) if three_sec_sum > 0 else None

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
        three_sec_views=three_sec_sum or None,
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
        roas=roas or 0.0,
        thumbstop_rate=thumb,
        profit=profit,
        poas=poas,
    )


def groupby_aggregate(
    rows: Iterable[Dict[str, Any]],
    group_key: str,
    cfg: Optional[MetricsConfig] = None,
    *,
    cogs_per_purchase: Optional[float] = None,
    smoothing_epsilon: Optional[float] = None,
) -> Dict[Any, Metrics]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        k = r.get(group_key)
        groups.setdefault(k, []).append(r)
    return {
        k: aggregate_rows(v, cfg, cogs_per_purchase=cogs_per_purchase, smoothing_epsilon=smoothing_epsilon)
        for k, v in groups.items()
    }


# -------------------- Quick accessors --------------------

def purchases_count(row: Dict[str, Any], cfg: Optional[MetricsConfig] = None) -> float:
    cfg = cfg or MetricsConfig()
    return _scan_actions(row, cfg.action_aliases["purchase"], cfg.window_keys)


def atc_count(row: Dict[str, Any], cfg: Optional[MetricsConfig] = None) -> float:
    cfg = cfg or MetricsConfig()
    return _scan_actions(row, cfg.action_aliases["add_to_cart"], cfg.window_keys)


def ctr_basic(row: Dict[str, Any], epsilon: float = 0.0) -> float:
    imps = _to_float(row.get("impressions"))
    clicks = _to_float(row.get("clicks"))
    if imps > 0:
        return clicks / imps
    return clicks / (imps + epsilon) if epsilon > 0 else 0.0


def cpa_basic(row: Dict[str, Any], cfg: Optional[MetricsConfig] = None) -> Optional[float]:
    cfg = cfg or MetricsConfig()
    spend = _to_float(row.get("spend"))
    p = purchases_count(row, cfg)
    return _safe_div(spend, p, None) if p > 0 else None


def roas_basic(row: Dict[str, Any], cfg: Optional[MetricsConfig] = None) -> float:
    cfg = cfg or MetricsConfig()
    spend = _to_float(row.get("spend"))
    rev = _scan_values(row, cfg.value_aliases["purchase"], cfg.value_keys)
    return _safe_div(rev, spend, 0.0) if spend > 0 else 0.0


# -------------------- Diagnostics --------------------

def explain_row(row: Dict[str, Any], cfg: Optional[MetricsConfig] = None) -> Dict[str, Any]:
    m, d = metrics_from_row(row, cfg, compute_diagnostics=True)  # type: ignore[misc]
    summary = m.as_dict()
    summary["_diagnostics"] = {
        "used_action_aliases": d.used_action_aliases,
        "used_value_aliases": d.used_value_aliases,
        "missing_required": d.missing_required,
        "anomalies": d.anomalies,
    }
    return summary


__all__ = [
    "MetricsConfig",
    "Metrics",
    "Diagnostics",
    "metrics_from_row",
    "aggregate_rows",
    "groupby_aggregate",
    "purchases_count",
    "atc_count",
    "ctr_basic",
    "cpa_basic",
    "roas_basic",
    "explain_row",
    # NEW exports
    "convert",
    "usd_to_account",
    "tripwire_threshold_account",
]
