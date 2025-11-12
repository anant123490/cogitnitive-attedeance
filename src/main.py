"""
Command line entry point for the attendance system.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .tracker import build_tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-faceted Image Recognition Attendance System")
    parser.add_argument("--input-dir", type=Path, help="Directory with images to process")
    parser.add_argument("--realtime", action="store_true", help="Enable realtime webcam processing")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for realtime mode")
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=30.0,
        help="Cooldown period before re-logging the same person in realtime mode",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process every N-th frame in realtime mode to reduce load",
    )
    parser.add_argument("--display", action="store_true", help="Display annotated video feed in realtime mode")
    parser.add_argument(
        "--repository",
        choices=["csv", "sqlite"],
        default="csv",
        help="Storage backend for attendance records",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        default=Path("data/attendance_events.csv"),
        help="Path to write detailed attendance events",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/attendance_summary.csv"),
        help="Path to write present/absent summary",
    )
    parser.add_argument("--encodings", type=Path, default=None, help="Path to known face encodings JSON file")
    parser.add_argument(
        "--roster",
        type=Path,
        default=None,
        help="Optional CSV roster with a 'person_id' column. Defaults to encodings file roster.",
    )
    parser.add_argument("--evidence-dir", type=Path, default=None, help="Directory for storing evidence snapshots")
    parser.add_argument("--enable-yolo", action="store_true", help="Enable YOLO crowd analysis")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config = load_config(
        known_embeddings_path=str(args.encodings) if args.encodings else None,
        output_dir=str(args.event_log.parent),
        evidence_dir=str(args.evidence_dir) if args.evidence_dir else None,
    )
    config.enable_object_detection = args.enable_yolo

    tracker = build_tracker(config, repository_target=args.repository, repository_path=args.event_log)

    from .session import AttendanceSession

    if args.roster:
        session = AttendanceSession.from_roster_file(args.roster)
    else:
        session = AttendanceSession.from_known_ids(tracker.known_person_ids())

    if not session.records:
        raise ValueError("Roster is empty. Provide known faces or a roster file.")

    try:
        if args.event_log.exists():
            args.event_log.unlink()
        if args.summary.exists():
            args.summary.unlink()

        if args.realtime:
            from .realtime import RealtimeAttendanceSystem

            system = RealtimeAttendanceSystem(
                tracker=tracker,
                session=session,
            summary_path=args.summary,
                camera_index=args.camera_index,
                frame_skip=args.frame_skip,
                cooldown_seconds=args.cooldown_seconds,
                display=args.display,
                save_evidence=True,
            )
            system.run()
        else:
            if not args.input_dir:
                raise ValueError("--input-dir is required in batch mode")
            image_paths = sorted(
                [p for p in args.input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            )
            total = 0
            for image_path in image_paths:
                events = tracker.process_path(image_path)
                session.ingest(events)
                total += len(events)
            logging.info("Processing complete: %d events logged.", total)
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Finalizing session…")
    finally:
        session.ensure_absentees()
        session.save(args.summary)
        logging.info("Attendance summary written to %s", args.summary)


if __name__ == "__main__":
    main()


