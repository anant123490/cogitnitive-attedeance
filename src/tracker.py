"""
Core attendance tracking orchestration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from .config import AttendanceConfig
from .pipelines import CrowdAnalyzer, ImagePreprocessor, InferencePipeline
from .recognizer import FaceRecognizer
from .storage import AttendanceEvent, AttendanceRepository, get_repository

LOGGER = logging.getLogger(__name__)


class AttendanceTracker:
    """Coordinates the data flow between pipelines and storage."""

    def __init__(self, config: AttendanceConfig, repository: AttendanceRepository) -> None:
        self.config = config
        self.repository = repository
        self.recognizer = FaceRecognizer(threshold=config.recognition_threshold)
        self.preprocessor = ImagePreprocessor(tolerated_blur=config.tolerated_blur)
        crowd = CrowdAnalyzer(config.yolo_weights_path) if config.enable_object_detection else None
        self.pipeline = InferencePipeline(self.preprocessor, self.recognizer, crowd_analyzer=crowd)

    def bootstrap(self) -> None:
        LOGGER.debug("Bootstrapping tracker with config: %s", self.config)
        self.recognizer.load_known_faces(self.config.known_embeddings_path)
        self.config.ensure_directories()

    def _create_events(
        self, matches: Iterable[Tuple[str, float, dict]], source: Union[Path, str]
    ) -> List[AttendanceEvent]:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        events: List[AttendanceEvent] = []
        for match in matches:
            person_id, confidence, metadata = match
            events.append(
                AttendanceEvent(
                    person_id=person_id,
                    confidence=confidence,
                    timestamp=timestamp,
                    source_path=str(source),
                    extra={**metadata},
                )
            )
        return events

    def process_path(self, path: Path) -> List[AttendanceEvent]:
        matches = self.pipeline.run(path)
        return self.log_matches(matches, path)

    def process_directory(self, directory: Path) -> int:
        files = [p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        total_events = 0
        for file_path in files:
            events = self.process_path(file_path)
            total_events += len(events)
        LOGGER.info("Total events logged: %d", total_events)
        return total_events

    def process_frame(self, frame: np.ndarray, source: str = "camera") -> List[AttendanceEvent]:
        matches = self.pipeline.run_on_array(frame)
        return self.log_matches(matches, source)

    def log_matches(
        self, matches: Sequence[Tuple[str, float, dict]], source: Union[Path, str]
    ) -> List[AttendanceEvent]:
        if not matches:
            return []
        events = self._create_events(matches, source)
        self.repository.save_many(events)
        LOGGER.info("Logged %d events from %s", len(events), source)
        return events

    def known_person_ids(self) -> List[str]:
        return [face.person_id for face in self.recognizer.known_faces]


def build_tracker(config: AttendanceConfig, repository_target: str, repository_path: Path) -> AttendanceTracker:
    repository = get_repository(repository_target, repository_path)
    tracker = AttendanceTracker(config, repository)
    tracker.bootstrap()
    return tracker


