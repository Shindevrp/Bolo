from __future__ import annotations

import time
from contextlib import contextmanager

from utils.logger import get_logger

logger = get_logger("timers")


@contextmanager
def timed_block(label: str):
    """Measure a block of code execution time with structured logging."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"block {label}", extra={"elapsed_s": round(elapsed, 4)})
