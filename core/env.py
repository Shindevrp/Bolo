"""Environment variable access for Bolo.

Configuration is read as ``BOLO_*``. For backwards compatibility the legacy
``TASA_*`` name is used as a fallback, so an existing ``.env`` or a Docker
deployment that still sets the old names keeps working unchanged.
"""

from __future__ import annotations

import os

__all__ = ["env", "env_bool", "env_float", "env_int", "env_list", "LEGACY_PREFIX"]

LEGACY_PREFIX = "TASA_"


def _name(key: str) -> str:
    return key if key.startswith("BOLO_") else f"BOLO_{key}"


def env(key: str, default: str = "") -> str:
    """Return ``BOLO_<key>``, else the legacy ``TASA_<key>``, else *default*.

    Presence -- not truthiness -- decides, so setting a variable to the empty
    string still overrides the default. Several settings use ``""`` as a
    meaningful value (an empty shadow-log path disables disk writes).
    """
    bolo = _name(key)
    if bolo in os.environ:
        return os.environ[bolo].strip()
    legacy = f"{LEGACY_PREFIX}{key}"
    if legacy in os.environ:
        return os.environ[legacy].strip()
    return default


def env_float(key: str, default: float) -> float:
    try:
        return float(env(key))
    except (TypeError, ValueError):
        return default


def env_int(key: str, default: int) -> int:
    try:
        return int(env(key))
    except (TypeError, ValueError):
        return default


def env_bool(key: str, default: bool) -> bool:
    value = env(key).lower()
    if not value:
        return default
    return value not in ("0", "false", "no", "off")


def env_list(key: str, default: str = "") -> list[str]:
    return [item.strip() for item in env(key, default).split(",") if item.strip()]
