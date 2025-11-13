"""
Configuration for Dean system
Constants and validation functions
"""

import logging
from typing import Any, Dict, Final, Iterable

logger: Final = logging.getLogger(__name__)

# =====================================================
# CONSTANTS
# =====================================================

# Database constraints
DB_NUMERIC_MIN: Final[float] = -9.9999
DB_NUMERIC_MAX: Final[float] = 9.9999
DB_CTR_MAX: Final[float] = 99.9999
DB_CPC_MAX: Final[float] = 9999.9999
DB_CPM_MAX: Final[float] = 9999.9999

# Supabase quirks
# Supabase production constraint currently accepts only the legacy 'testing' stage
# for the creative_performance table. Use this canonical value for reads/writes
# while continuing to treat the logical stage as ASC+ elsewhere in the system.
CREATIVE_PERFORMANCE_STAGE_VALUE: Final[str] = "testing"

# Budget constraints
ASC_PLUS_BUDGET_MIN: Final[float] = 5.0
ASC_PLUS_BUDGET_MAX: Final[float] = 500.0
ASC_PLUS_MIN_BUDGET_PER_CREATIVE: Final[float] = 5.0

# Performance thresholds
CTR_QUALITY_THRESHOLD: Final[float] = 2.0
CTR_STABILITY_THRESHOLD: Final[float] = 1.0
CPA_QUALITY_THRESHOLD: Final[float] = 30.0
CPA_MAX_THRESHOLD: Final[float] = 60.0
ROAS_MIN_THRESHOLD: Final[float] = 1.0

# Performance score weights
PERFORMANCE_SCORE_BASE: Final[float] = 0.5
PERFORMANCE_SCORE_CTR_MAX: Final[float] = 0.3
PERFORMANCE_SCORE_CPA_MAX: Final[float] = 0.2
PERFORMANCE_SCORE_ROAS_MAX: Final[float] = 0.2

# Ad age limits
MAX_AD_AGE_DAYS: Final[int] = 365
MAX_STAGE_DURATION_HOURS: Final[int] = 168  # 1 week

# Flux credit cache
FLUX_CREDITS_CACHE_TTL_SECONDS: Final[int] = 60

# Default values
DEFAULT_SAFE_FLOAT_MAX: Final[float] = 999999.99
DB_ROAS_MAX: Final[float] = 9999.9999
DB_CPA_MAX: Final[float] = 9999.9999
DB_DWELL_TIME_MAX: Final[float] = 999999.99
DB_FREQUENCY_MAX: Final[float] = 999.99
DB_RATE_MAX: Final[float] = 9.9999
DB_GLOBAL_FLOAT_MAX: Final[float] = 999999999.99

# =====================================================
# VALIDATION
# =====================================================

def _find_duplicate_keys(cfg: Dict[str, Any]) -> Iterable[str]:
    """
    Detect obvious duplicate ASC+ sections after config merge.
    We treat any top-level key that starts with 'asc_plus' (besides the canonical
    'asc_plus' mapping) as a duplicate.
    """
    for key in cfg.keys():
        if key != "asc_plus" and key.startswith("asc_plus"):
            yield key


def validate_asc_plus_config(cfg: Dict[str, Any]) -> None:
    """
    Sanity-check merged configuration for the ASC+ stage.

    - Ensures a single `asc_plus` mapping exists.
    - Warns (and sanitises) redundant placement keys when Advantage+ placements are enabled.
    - Raises when duplicate sections are detected.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Settings payload must be a dictionary.")

    asc_plus_cfg = cfg.get("asc_plus")
    if not isinstance(asc_plus_cfg, dict):
        raise ValueError("Missing or invalid `asc_plus` configuration. Expected a mapping.")

    duplicates = list(_find_duplicate_keys(cfg))
    if duplicates:
        raise ValueError(
            f"Duplicate ASC+ configuration detected in keys: {', '.join(sorted(duplicates))}"
        )

    placements_cfg = cfg.get("placements") or {}
    asc_plus_placements = placements_cfg.get("asc_plus")
    if isinstance(asc_plus_placements, dict):
        if asc_plus_placements.get("advantage_plus_placements"):
            if asc_plus_placements.get("publisher_platforms"):
                logger.warning(
                    "Ignoring placements.asc_plus.publisher_platforms because "
                    "advantage_plus_placements=true for ASC+. Meta will manage placements automatically."
                )
                asc_plus_placements.pop("publisher_platforms", None)
        placements_cfg["asc_plus"] = asc_plus_placements
        cfg["placements"] = placements_cfg

