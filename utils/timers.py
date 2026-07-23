from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed_block(label: str):
    """Measure a block of code execution time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")
