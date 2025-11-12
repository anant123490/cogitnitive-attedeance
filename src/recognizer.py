"""
Face recognition utilities powered by `facenet-pytorch`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class KnownFace:
    """Data structure for known individuals."""

    person_id: str
    embedding: np.ndarray
    metadata: Dict[str, str]


class FaceRecognizer:
    """Encapsulates face detection, encoding, and matching operations."""

    def __init__(self, threshold: float = 0.45, device: str | None = None) -> None:
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(keep_all=True, device=self.device)
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.known_faces: List[KnownFace] = []

    def load_known_faces(self, encodings_path: Path) -> None:
        """Load known face embeddings from a JSON file."""
        if not encodings_path.exists():
            LOGGER.warning("No encodings file found at %s", encodings_path)
            return

        content = json.loads(encodings_path.read_text(encoding="utf-8"))
        self.known_faces = [
            KnownFace(
                person_id=item["person_id"],
                embedding=self._normalize(np.asarray(item["embedding"], dtype=np.float32)),
                metadata=item.get("metadata", {}),
            )
            for item in content
        ]
        LOGGER.info("Loaded %d known faces", len(self.known_faces))

    def encode_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Compute face embeddings for the detected faces in the given RGB image.
        """
        pil_image = Image.fromarray(image)
        face_crops, probabilities = self.detector(pil_image, return_prob=True)
        if face_crops is None:
            return []

        valid_faces = [
            crop.to(self.device)
            for crop, prob in zip(face_crops, probabilities)
            if prob and prob > 0.90
        ]
        if not valid_faces:
            return []

        batch = torch.stack(valid_faces)
        with torch.inference_mode():
            embeddings = self.embedder(batch).cpu().numpy()
        return [self._normalize(embedding) for embedding in embeddings]

    def match_face(self, embedding: np.ndarray) -> Tuple[str, float, Dict[str, str]]:
        """
        Match a face embedding against known faces and return the best match.
        Distance metric: cosine distance (lower is better).
        """
        if not self.known_faces:
            return ("unknown", 1.0, {})

        distances = [
            self._cosine_distance(embedding, known.embedding) for known in self.known_faces
        ]
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance <= self.threshold:
            matched = self.known_faces[best_index]
            return (matched.person_id, best_distance, matched.metadata)
        return ("unknown", best_distance, {})

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 1.0
        cosine_similarity = float(np.dot(a, b) / denom)
        return 1.0 - cosine_similarity

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


