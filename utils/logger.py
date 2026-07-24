from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log, default=str)


def get_logger(name: str = "TASA", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(f"TASA.{name}" if name != "TASA" else "TASA")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = get_logger("TASA")
