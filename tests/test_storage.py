"""
Tests for storage backends.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.storage import AttendanceEvent, CSVAttendanceRepository


def test_csv_repository_writes_events(tmp_path: Path) -> None:
    target = tmp_path / "attendance.csv"
    repo = CSVAttendanceRepository(target)

    events = [
        AttendanceEvent(
            person_id="alice",
            confidence=0.12,
            timestamp="2025-11-12T10:00:00Z",
            source_path="image1.jpg",
            extra={"location": "HQ"},
        ),
        AttendanceEvent(
            person_id="bob",
            confidence=0.34,
            timestamp="2025-11-12T10:01:00Z",
            source_path="image2.jpg",
            extra={"location": "HQ"},
        ),
    ]

    repo.save_many(events)

    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["person_id"] == "alice"
    assert rows[1]["person_id"] == "bob"


