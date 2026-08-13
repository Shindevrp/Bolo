from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger("persistence")


class SessionPersistence:
    """Save and load session state to JSON files."""

    def __init__(self, storage_dir: str = "~/.tasa/sessions") -> None:
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        path = self.storage_dir / f"{session_id}.json"
        try:
            path.write_text(json.dumps(data, indent=2, default=str))
            logger.debug(f"session saved: {session_id}")
        except Exception as e:
            logger.warning(f"failed to save session {session_id}: {e}")

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self.storage_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"failed to load session {session_id}: {e}")
            return None

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        files = sorted(self.storage_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        results: list[dict[str, Any]] = []
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text())
                data["_file"] = f.name
                results.append(data)
            except Exception:
                continue
        return results

    def delete(self, session_id: str) -> bool:
        path = self.storage_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
