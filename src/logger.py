"""Structured event log output (JSON + CSV) and basic run logging."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from .events import Event


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("surveillance")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class EventLogger:
    def __init__(self):
        self.events: list[Event] = []

    def add(self, events: list[Event]):
        self.events.extend(events)

    def save(self, output_dir: Path, video_name: str, meta: dict):
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "source_video": video_name,
            "meta": meta,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }
        json_path = output_dir / "events.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        csv_path = output_dir / "events.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "timestamp_s", "track_id", "zone", "event_type", "bbox", "confidence"])
            for e in self.events:
                d = e.to_dict()
                writer.writerow([d["frame"], d["timestamp_s"], d["track_id"], d["zone"], d["event_type"], d["bbox"], d["confidence"]])

        return json_path, csv_path
