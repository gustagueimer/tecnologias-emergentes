from pathlib import Path
from typing import Any


class CaptureStore:
    def __init__(self, captures_dir: Path):
        self.captures_dir = captures_dir

    def list_captures(self, limit: int = 30) -> list[dict[str, Any]]:
        if not self.captures_dir.exists():
            return []

        files = sorted(
            self.captures_dir.glob("*.jpg"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        result: list[dict[str, Any]] = []
        for file_path in files[:limit]:
            result.append(
                {
                    "name": file_path.name,
                    "url": f"/static/captures/{file_path.name}",
                    "size": file_path.stat().st_size,
                }
            )

        return result