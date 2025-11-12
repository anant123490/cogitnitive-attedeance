"""
Attendance storage backends.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Protocol


@dataclass
class AttendanceEvent:
    """Normalized attendance event."""

    person_id: str
    confidence: float
    timestamp: str
    source_path: str
    extra: Dict[str, str]


class AttendanceRepository(Protocol):
    """Protocol for different storage backends."""

    def save_many(self, events: Iterable[AttendanceEvent]) -> None:
        ...


class CSVAttendanceRepository:
    """Persist attendance events to a CSV file."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def save_many(self, events: Iterable[AttendanceEvent]) -> None:
        rows = [asdict(event) for event in events]
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()) if rows else None
            if writer and handle.tell() == 0:
                writer.writeheader()
            if writer:
                for row in rows:
                    writer.writerow(row)


class SQLiteAttendanceRepository:
    """Persist attendance events to a SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    source_path TEXT,
                    extra TEXT
                )
                """
            )
            conn.commit()

    def save_many(self, events: Iterable[AttendanceEvent]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO attendance (person_id, confidence, timestamp, source_path, extra)
                VALUES (:person_id, :confidence, :timestamp, :source_path, :extra)
                """,
                [
                    {
                        "person_id": event.person_id,
                        "confidence": event.confidence,
                        "timestamp": event.timestamp,
                        "source_path": event.source_path,
                        "extra": str(event.extra),
                    }
                    for event in events
                ],
            )
            conn.commit()


def get_repository(target: str, path: Path) -> AttendanceRepository:
    """Factory to instantiate the appropriate repository."""
    if target == "csv":
        return CSVAttendanceRepository(path)
    if target == "sqlite":
        return SQLiteAttendanceRepository(path)
    raise ValueError(f"Unsupported repository target: {target}")


