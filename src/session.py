"""
Attendance session tracking utilities.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .storage import AttendanceEvent


@dataclass
class SessionRecord:
    person_id: str
    status: str = "absent"
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    confidence: Optional[float] = None


class AttendanceSession:
    """Track present/absent status for a roster during a session."""

    def __init__(self, roster: Iterable[str]) -> None:
        unique_ids = sorted({person_id for person_id in roster})
        self.records: Dict[str, SessionRecord] = {
            person_id: SessionRecord(person_id=person_id) for person_id in unique_ids
        }

    @classmethod
    def from_roster_file(cls, roster_path: Path) -> "AttendanceSession":
        if not roster_path.exists():
            raise FileNotFoundError(f"Roster file not found: {roster_path}")
        with roster_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "person_id" not in reader.fieldnames:
                raise ValueError("Roster file must contain a 'person_id' column.")
            roster = [row["person_id"].strip() for row in reader if row.get("person_id")]
        return cls(roster)

    @classmethod
    def from_known_ids(cls, person_ids: Iterable[str]) -> "AttendanceSession":
        return cls(person_ids)

    def ingest(self, events: Iterable[AttendanceEvent]) -> None:
        for event in events:
            if event.person_id.lower() == "unknown":
                continue
            self.mark_present(
                person_id=event.person_id,
                timestamp=event.timestamp,
                confidence=event.confidence,
            )

    def mark_present(self, person_id: str, timestamp: Optional[str], confidence: float) -> None:
        record = self.records.get(person_id)
        if record is None:
            record = SessionRecord(person_id=person_id)
            self.records[person_id] = record
        record.status = "present"
        record.confidence = confidence
        if timestamp:
            if record.first_seen is None:
                record.first_seen = timestamp
            record.last_seen = timestamp

    def save(self, output_path: Path, summary_time: Optional[str] = None) -> None:
        if summary_time is None:
            summary_time = datetime.now(tz=timezone.utc).isoformat()

        rows = []
        for person_id in sorted(self.records):
            record = self.records[person_id]
            if record.status == "present":
                timestamp = record.last_seen or record.first_seen or summary_time
            else:
                timestamp = summary_time
            rows.append(
                {
                    "Unique ID": record.person_id,
                    "Status": record.status.capitalize(),
                    "Timestamp": timestamp or "",
                    "First Seen": record.first_seen or "",
                    "Last Seen": record.last_seen or "",
                    "Confidence": f"{record.confidence:.4f}" if record.confidence is not None else "",
                }
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.DataFrame(rows)
            df.to_excel(output_path, index=False)
        else:
            fieldnames = ["Unique ID", "Status", "Timestamp", "First Seen", "Last Seen", "Confidence"]
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def ensure_absentees(self) -> None:
        """Ensure roster exists even if no events were ingested."""
        if not self.records:
            raise ValueError("No roster provided; cannot determine absentees.")
        # Nothing else needed; records are initialized to 'absent'.

