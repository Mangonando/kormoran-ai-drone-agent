"""Battery utilities for the safety supervisor."""

from __future__ import annotations

#: Minimum battery percentage before a forced landing is triggered.
RESERVE_PCT: float = 15.0


def is_low(battery_pct: float, reserve_pct: float = RESERVE_PCT) -> bool:
    """Return ``True`` when *battery_pct* is at or below *reserve_pct*."""
    return battery_pct <= reserve_pct
