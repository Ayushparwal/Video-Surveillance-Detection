"""Wires detection+tracking -> zone events -> visualization -> output.

Kept deliberately thin: this module's only job is orchestration and
frame I/O. All actual logic lives in the single-responsibility modules
it imports, so any piece (detector, event logic, renderer) can be
swapped or unit-tested independently.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .detector_tracker import PersonTracker
from .events import EventDetector
from .logger import EventLogger, setup_logging
from .visualizer import draw_detections, draw_hud, draw_zones
from .zones import load_zones


class SurveillancePipeline:
    def __init__(
        self,
        video_path: str,
        zones_path: str,
        output_dir: str,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        device: str | None = None,
        frame_stride: int = 1,
        no_video_output: bool = False,
        verbose: bool = False,
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.frame_stride = max(1, frame_stride)
        self.no_video_output = no_video_output

        self.logger = setup_logging(self.output_dir, verbose)
        self.zones = load_zones(zones_path)
        self.tracker = PersonTracker(model_path=model_path, conf_threshold=conf_threshold, device=device)
        self.event_detector = EventDetector(self.zones)
        self.event_logger = EventLogger()

    def run(self) -> dict:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Opened {self.video_path.name}: {width}x{height} @ {src_fps:.1f}fps, {total_frames} frames")
        self.logger.info(f"Loaded {len(self.zones)} zone(s): {[z.name for z in self.zones]}")

        writer = None
        out_video_path = self.output_dir / f"{self.video_path.stem}_annotated.mp4"
        if not self.no_video_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, src_fps / self.frame_stride, (width, height))

        frame_idx = 0
        processed = 0
        empty_frame_count = 0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_stride != 0:
                frame_idx += 1
                continue

            if frame is None or frame.size == 0:
                # Corrupt/empty frame -- log and skip rather than crash the run.
                empty_frame_count += 1
                self.logger.warning(f"Empty/corrupt frame at index {frame_idx}, skipping")
                frame_idx += 1
                continue

            timestamp = frame_idx / src_fps

            try:
                detections = self.tracker.track_frame(frame)
            except Exception as exc:
                # A single bad frame (e.g. transient decode glitch) shouldn't
                # kill an hours-long run.
                self.logger.error(f"Detection failed on frame {frame_idx}: {exc}")
                detections = []

            new_events = self.event_detector.update(frame_idx, timestamp, detections)
            if new_events:
                self.event_logger.add(new_events)
                for e in new_events:
                    self.logger.info(f"EVENT {e.event_type.upper()} | track={e.track_id} zone={e.zone_name} t={e.timestamp:.2f}s")

            if writer is not None:
                alert_ids = {e.track_id for e in new_events}
                annotated = draw_zones(frame.copy(), self.zones)
                annotated = draw_detections(annotated, detections, alert_ids)
                elapsed = time.time() - t_start
                live_fps = processed / elapsed if elapsed > 0 else 0.0
                annotated = draw_hud(annotated, frame_idx, live_fps, len(self.event_logger.events))
                writer.write(annotated)

            processed += 1
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        elapsed = time.time() - t_start
        avg_fps = processed / elapsed if elapsed > 0 else 0.0

        meta = {
            "processed_frames": processed,
            "total_frames": total_frames,
            "frame_stride": self.frame_stride,
            "source_fps": src_fps,
            "resolution": [width, height],
            "wall_clock_seconds": round(elapsed, 2),
            "avg_processing_fps": round(avg_fps, 2),
            "empty_frames_skipped": empty_frame_count,
        }
        json_path, csv_path = self.event_logger.save(self.output_dir, self.video_path.name, meta)

        self.logger.info(f"Done. {len(self.event_logger.events)} events in {processed} frames ({avg_fps:.1f} fps avg)")
        self.logger.info(f"Event log: {json_path}")
        if writer is not None:
            self.logger.info(f"Annotated video: {out_video_path}")

        return {
            "events_path": str(json_path),
            "csv_path": str(csv_path),
            "video_path": str(out_video_path) if writer is not None else None,
            "meta": meta,
            "event_count": len(self.event_logger.events),
        }
