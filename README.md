# Multi-Faceted Image Recognition Attendance System

## Overview
This project demonstrates a modular attendance system that combines face recognition, object detection, and contextual metadata to mark attendance events. It aims to:
- Detect and recognize individuals in images or camera frames.
- Validate contextual cues (e.g., location, timestamp) to reduce false positives.
- Store attendance logs with evidence snapshots for auditing.

## Features
- Embedding-based face recognition using `facenet-pytorch` (MTCNN + InceptionResnetV1).
- Optional YOLOv5 object detection for crowd density estimation.
- Pre-processing pipeline with alignment and quality checks.
- Pluggable storage backends (`CSV`, `SQLite`) via a repository pattern.
- CLI utility to process image directories or capture from a webcam.
- Unit-test-ready architecture with dependency injection.

## Project Structure
```
attendance_system/
├── data/                 # Sample data and embeddings
├── models/               # Serialized models (face encodings, YOLO weights)
├── notebooks/            # Experiments and prototyping
├── src/
│   ├── config.py         # App configuration and paths
│   ├── pipelines.py      # Pre-processing and inference pipeline
│   ├── recognizer.py     # Face recognition utilities
│   ├── realtime.py       # Webcam-based realtime attendance
│   ├── session.py        # Session roster management
│   ├── storage.py        # Attendance repositories
│   ├── tracker.py        # Core attendance tracker logic
│   └── main.py           # CLI entry point
├── tests/
│   ├── __init__.py
│   └── test_storage.py
├── requirements.txt
├── setup.cfg             # Tooling config (linting, formatting)
└── README.md
```

## Quick Start
1. **Use a compatible Python version**
   - Python 3.13+ on Windows is supported (PyTorch and facenet-pytorch ship wheels).

2. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the sample pipeline**
   ```powershell
   python -m src.main --input-dir data/samples --summary data/attendance_summary.csv --event-log data/attendance_events.csv
   ```

5. **Run realtime webcam mode**
   ```powershell
   python -m src.main --realtime --camera-index 0 --display --summary data/attendance_summary.csv
   ```
   Press `q` to stop; recognized faces are logged to the configured repository with optional evidence frames saved under `data/evidence`.

## Preparing the Dataset
- Place one folder per learner under `data/known_faces/` (e.g. `data/known_faces/alice/alice1.jpg`).
- Generate encodings:
  ```powershell
  python -m src.build_encodings --dataset-dir data/known_faces --output-path models/encodings.json
  ```
- The realtime or batch commands will compare live detections against this roster. Any name absent from the dataset remains marked `absent` in the summary CSV.

## Notes
- Place pre-computed face encodings in `models/encodings.json` (`python -m src.build_encodings` can generate it).
- YOLO weights (optional) should go into `models/yolov5s.pt`.
- Customize configuration via environment variables or `config.py`.