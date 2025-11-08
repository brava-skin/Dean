import logging
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)


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

