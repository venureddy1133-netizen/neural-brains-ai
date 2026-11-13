"""SQLite-backed experience memory."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS experiences ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "text TEXT NOT NULL,label TEXT NOT NULL,"
                "source TEXT NOT NULL DEFAULT 'user',created_at TEXT NOT NULL)"
            )
            db.commit()

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def remember(self, text, label, source="user"):
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO experiences(text,label,source,created_at) VALUES(?,?,?,?)",
                (text, label, source, datetime.now(timezone.utc).isoformat()),
            )
            db.commit()

    def count(self):
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def recent(self, limit=10):
        with closing(self._connect()) as db:
            return [
                dict(row) for row in db.execute(
                    "SELECT id,text,label,source,created_at FROM experiences "
                    "ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            ]

    def recall(self, text, limit=3):
        words = {word.lower() for word in text.split() if len(word) > 2}
        scored = []
        for row in self.recent(200):
            overlap = len(words & {word.lower() for word in row["text"].split() if len(word) > 2})
            if overlap:
                scored.append((overlap, row))
        scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
        return [row for _, row in scored[:limit]]
