"""
Pre-processing and inference pipelines for the attendance system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .recognizer import FaceRecognizer

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Result of preprocessing an image."""

    image: np.ndarray
    quality_score: float
    is_valid: bool


class ImagePreprocessor:
    """Handles loading and validating image frames."""

    def __init__(self, tolerated_blur: float) -> None:
        self.tolerated_blur = tolerated_blur

    def load_image(self, path: Path) -> Optional[np.ndarray]:
        image = cv2.imread(str(path))
        if image is None:
            LOGGER.error("Failed to read image at %s", path)
            return None
        # Convert to RGB for compatibility with face_recognition
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def evaluate_quality(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def process(self, path: Path) -> Optional[PreprocessResult]:
        image = self.load_image(path)
        if image is None:
            return None
        quality = self.evaluate_quality(image)
        is_valid = quality >= self.tolerated_blur
        return PreprocessResult(image=image, quality_score=quality, is_valid=is_valid)

    def process_frame(self, image: np.ndarray) -> PreprocessResult:
        """Validate an already-loaded RGB frame."""
        quality = self.evaluate_quality(image)
        is_valid = quality >= self.tolerated_blur
        return PreprocessResult(image=image, quality_score=quality, is_valid=is_valid)


class CrowdAnalyzer:
    """Optional YOLO-based crowd analyzer."""

    def __init__(self, weights_path: Optional[Path]) -> None:
        self.model = None
        if weights_path and weights_path.exists():
            try:
                self.model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights_path), force_reload=False)
                LOGGER.info("Loaded YOLO model from %s", weights_path)
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.warning("Failed to load YOLO weights: %s", exc)

    def estimate(self, image: np.ndarray) -> int:
        if self.model is None:
            return 0
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()
        return int(detections.shape[0])


class InferencePipeline:
    """Runs the full inference pipeline for a single image."""

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        recognizer: FaceRecognizer,
        crowd_analyzer: Optional[CrowdAnalyzer] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.recognizer = recognizer
        self.crowd_analyzer = crowd_analyzer

    def run(self, image_path: Path) -> List[Tuple[str, float, dict]]:
        preprocessed = self.preprocessor.process(image_path)
        if preprocessed is None or not preprocessed.is_valid:
            LOGGER.debug("Skipping %s due to insufficient quality", image_path)
            return []

        embeddings = self.recognizer.encode_image(preprocessed.image)
        matches: List[Tuple[str, float, dict]] = []
        for embedding in embeddings:
            matches.append(self.recognizer.match_face(embedding))
        return matches

    def run_on_array(self, image: np.ndarray) -> List[Tuple[str, float, dict]]:
        """
        Execute inference on an in-memory RGB frame.
        """
        preprocessed = self.preprocessor.process_frame(image)
        if not preprocessed.is_valid:
            return []

        embeddings = self.recognizer.encode_image(preprocessed.image)
        return [self.recognizer.match_face(embedding) for embedding in embeddings]


