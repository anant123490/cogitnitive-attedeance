"""
Utility script to generate face encodings JSON from a labeled dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from .recognizer import FaceRecognizer

LOGGER = logging.getLogger(__name__)


def collect_image_paths(dataset_dir: Path) -> Dict[str, List[Path]]:
    """Map of person_id to image paths."""
    mapping: Dict[str, List[Path]] = {}
    for person_dir in dataset_dir.iterdir():
        if person_dir.is_dir():
            images = [
                path
                for path in person_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if images:
                mapping[person_dir.name] = images
    return mapping


def generate_encodings(dataset_dir: Path, recognizer: FaceRecognizer) -> List[dict]:
    """Generate embeddings for each image in the dataset."""
    data: List[dict] = []
    for person_id, paths in collect_image_paths(dataset_dir).items():
        for image_path in paths:
            try:
                image = _load_image(image_path)
            except ValueError as exc:
                LOGGER.warning("Failed to load %s: %s", image_path, exc)
                continue

            encodings = recognizer.encode_image(image)
            if not encodings:
                LOGGER.warning("No face found in %s; skipping.", image_path)
                continue
            data.append(
                {
                    "person_id": person_id,
                    "embedding": encodings[0].astype(np.float32).tolist(),
                    "metadata": {"source": image_path.name},
                }
            )
            LOGGER.info("Encoded %s from %s", person_id, image_path.name)
    return data


def _load_image(image_path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Unable to read image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_encodings(encodings: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(encodings, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d encodings to %s", len(encodings), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build face encodings JSON.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/known_faces"),
        help="Directory structured as person_id/ *.jpg images",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/encodings.json"),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    recognizer = FaceRecognizer(threshold=0.35)
    encodings = generate_encodings(args.dataset_dir, recognizer)
    if not encodings:
        LOGGER.warning("No encodings generated. Check your dataset.")
    save_encodings(encodings, args.output_path)


if __name__ == "__main__":
    main()


