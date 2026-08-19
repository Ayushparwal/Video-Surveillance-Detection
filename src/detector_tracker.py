"""
Detection + tracking stage.

Design decision: we use Ultralytics YOLOv8's built-in `.track()` rather
than wiring up a separate DeepSORT/ByteTrack library by hand. As of
ultralytics>=8.1 this runs ByteTrack (or BoT-SORT) internally and keeps
detection + association in one optimized call, which is both simpler
and faster than a hand-rolled two-stage pipeline for this scope. See
README "Model Choices" for the full trade-off discussion (why YOLOv8
over Faster R-CNN, why ByteTrack over DeepSORT).

This module only knows about "detections with track IDs" -- it has zero
knowledge of zones or events, so it can be swapped for a different
detector/tracker without touching downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class index for "person"


@dataclass
class Detection:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of the box -- used for zone tests (see zones.py)."""
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class PersonTracker:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        tracker_cfg: str = "bytetrack.yaml",
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker_cfg = tracker_cfg
        self.device = device

    def track_frame(self, frame: np.ndarray) -> list[Detection]:
        """Run detection+tracking on a single frame. `persist=True` tells
        ultralytics to keep the tracker's internal state (track ID history,
        Kalman filters, etc.) alive across calls -- without this every
        frame would be treated as a fresh video and IDs would never
        persist, which defeats the point of tracking."""
        results = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=self.conf_threshold,
            tracker=self.tracker_cfg,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            # No tracks this frame -- e.g. empty frame or everything below
            # confidence threshold. Not an error, just nothing to report.
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), tid, conf in zip(boxes, ids, confs):
            detections.append(Detection(track_id=int(tid), x1=x1, y1=y1, x2=x2, y2=y2, conf=float(conf)))
        return detections

    def reset(self):
        """Clear tracker state. Call between unrelated videos/clips so
        track IDs (and the ByteTrack Kalman state) from clip A don't leak
        into clip B."""
        self.model.predictor = None
