"""
Configuration utilities for the multi-faceted attendance system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AttendanceConfig:
    """Runtime configuration for the attendance system."""

    known_embeddings_path: Path = Path("models/encodings.json")
    yolo_weights_path: Optional[Path] = Path("models/yolov5s.pt")
    output_dir: Path = Path("data/outputs")
    evidence_dir: Path = Path("data/evidence")
    tolerated_blur: float = 100.0
    recognition_threshold: float = 0.35
    enable_object_detection: bool = False
    crowd_threshold: int = 10
    allowed_locations: List[str] = field(default_factory=lambda: ["default"])

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)


def load_config(
    known_embeddings_path: Optional[str] = None,
    yolo_weights_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    evidence_dir: Optional[str] = None,
) -> AttendanceConfig:
    """
    Load configuration with optional overrides.
    """
    cfg = AttendanceConfig()
    if known_embeddings_path:
        cfg.known_embeddings_path = Path(known_embeddings_path)
    if yolo_weights_path:
        cfg.yolo_weights_path = Path(yolo_weights_path)
    if output_dir:
        cfg.output_dir = Path(output_dir)
    if evidence_dir:
        cfg.evidence_dir = Path(evidence_dir)
    cfg.ensure_directories()
    return cfg


