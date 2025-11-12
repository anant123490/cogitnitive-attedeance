"""
Realtime attendance capture utilities.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .session import AttendanceSession
from .tracker import AttendanceEvent, AttendanceTracker

LOGGER = logging.getLogger(__name__)


class RealtimeAttendanceSystem:
    """
    Capture frames from a video source, recognize faces, and log attendance events.
    """

    def __init__(
        self,
        tracker: AttendanceTracker,
        session: AttendanceSession,
        summary_path: Path,
        camera_index: int = 0,
        frame_skip: int = 2,
        cooldown_seconds: float = 30.0,
        display: bool = False,
        save_evidence: bool = True,
    ) -> None:
        self.tracker = tracker
        self.session = session
        self.summary_path = summary_path
        self.camera_index = camera_index
        self.frame_skip = max(1, frame_skip)
        self.cooldown_seconds = cooldown_seconds
        self.display = display
        self.save_evidence = save_evidence
        self.last_seen: Dict[str, float] = {}

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.camera_index}")

        LOGGER.info("Realtime attendance started on camera index %s", self.camera_index)
        frame_counter = 0
        self.session.ensure_absentees()
        self._write_summary()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    LOGGER.warning("Failed to read frame from camera; stopping.")
                    break

                frame_counter += 1
                if frame_counter % self.frame_skip != 0:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                matches = self.tracker.pipeline.run_on_array(rgb_frame)
                recognized = self._filter_matches(matches)

                if recognized:
                    events = self.tracker.log_matches(
                        recognized, source=f"camera:{self.camera_index}"
                    )
                    self.session.ingest(events)
                    if self.save_evidence:
                        self._persist_evidence(frame, events)

                self._write_summary()

                if self.display and not self._render(frame, recognized):
                    break
        finally:
            cap.release()
            if self.display:
                cv2.destroyAllWindows()
            LOGGER.info("Realtime attendance stopped.")

    def _filter_matches(
        self, matches: Iterable[Tuple[str, float, dict]]
    ) -> List[Tuple[str, float, dict]]:
        now = time.time()
        filtered: List[Tuple[str, float, dict]] = []
        for person_id, distance, metadata in matches:
            if person_id == "unknown":
                continue
            last_seen = self.last_seen.get(person_id, 0.0)
            if now - last_seen < self.cooldown_seconds:
                continue
            filtered.append((person_id, distance, metadata))
            self.last_seen[person_id] = now
        return filtered

    def _render(self, frame: np.ndarray, matches: Iterable[Tuple[str, float, dict]]) -> bool:
        annotated = frame.copy()
        names = [match[0] for match in matches] or ["N/A"]
        label = f"Recognized: {', '.join(names)}"
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Realtime Attendance", annotated)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def _persist_evidence(self, frame: np.ndarray, events: Iterable[AttendanceEvent]) -> None:
        if not events:
            return
        evidence_dir = self.tracker.config.evidence_dir
        evidence_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        for event in events:
            filename = f"{timestamp}_{event.person_id.replace(' ', '_')}.jpg"
            target_path = evidence_dir / filename
            cv2.imwrite(str(target_path), frame)
            LOGGER.debug("Saved evidence frame to %s", target_path)

    def _write_summary(self) -> None:
        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        self.session.save(self.summary_path, summary_time=timestamp)

