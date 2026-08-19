"""Drawing utilities for the annotated output video."""

from __future__ import annotations

import cv2
import numpy as np

from .detector_tracker import Detection
from .zones import Zone

TRACK_COLOR = (60, 220, 60)
ALERT_COLOR = (0, 0, 255)


def draw_zones(frame: np.ndarray, zones: list[Zone], alpha: float = 0.18) -> np.ndarray:
    overlay = frame.copy()
    for zone in zones:
        pts = np.array(zone.points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], zone.color)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    for zone in zones:
        pts = np.array(zone.points, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=zone.color, thickness=2)
        label_pt = pts[0]
        cv2.putText(
            frame, f"{zone.name} ({zone.type})", (label_pt[0], max(15, label_pt[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color, 2, cv2.LINE_AA,
        )
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    alert_track_ids: set[int] | None = None,
) -> np.ndarray:
    alert_track_ids = alert_track_ids or set()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = ALERT_COLOR if det.track_id in alert_track_ids else TRACK_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {det.track_id}  {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # Foot point marker -- the actual point used for zone tests, drawn
        # so it's obvious during debugging/demo why a zone did or didn't trigger.
        fx, fy = det.foot_point
        cv2.circle(frame, (int(fx), int(fy)), 4, color, -1)
    return frame


def draw_hud(frame: np.ndarray, frame_idx: int, fps: float, active_events: int) -> np.ndarray:
    text = f"frame {frame_idx}  |  {fps:.1f} fps  |  events this run: {active_events}"
    cv2.putText(frame, text, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
