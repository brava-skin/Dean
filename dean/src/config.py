import logging
from typing import Any, Dict, Final, Iterable

logger: Final = logging.getLogger(__name__)

DB_CPC_MAX: Final[float] = 9999.9999
DB_CPM_MAX: Final[float] = 9999.9999
DB_ROAS_MAX: Final[float] = 9999.9999
DB_CPA_MAX: Final[float] = 9999.9999

CREATIVE_PERFORMANCE_STAGE_VALUE: Final[str] = "testing"

ASC_PLUS_BUDGET_MIN: Final[float] = 5.0
ASC_PLUS_BUDGET_MAX: Final[float] = 500.0
ASC_PLUS_MIN_BUDGET_PER_CREATIVE: Final[float] = 5.0

MAX_AD_AGE_DAYS: Final[int] = 365

FLUX_CREDITS_CACHE_TTL_SECONDS: Final[int] = 60

DEFAULT_SAFE_FLOAT_MAX: Final[float] = 999999.99


def _find_duplicate_keys(cfg: Dict[str, Any]) -> Iterable[str]:
    for key in cfg.keys():
        if key != "asc_plus" and key.startswith("asc_plus"):
            yield key


def validate_asc_plus_config(cfg: Dict[str, Any]) -> None:
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

