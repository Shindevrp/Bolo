from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreConfig:
    """Basic configuration for the core layer."""

    default_timeout: float = 30.0
    default_language: str = "en"
