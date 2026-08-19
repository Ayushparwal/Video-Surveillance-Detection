"""
Event detection logic.

This is the part of the assignment with real design decisions in it, so
the reasoning is documented inline rather than just in the README.

State model
-----------
For every (track_id, zone_name) pair we keep a small state record:

    {
        "inside": bool,           # is this track currently inside this zone?
        "entered_at": float|None, # timestamp (s) of the most recent entry
        "intrusion_fired": bool,  # have we already logged an intrusion for
                                   # this continuous dwell? (dedup)
        "loiter_fired": bool,     # have we already logged loitering for
                                   # this continuous dwell? (dedup)
        "last_seen_frame": int,   # for pruning stale tracks
    }

Edge-triggered vs level-triggered
----------------------------------
- Intrusion is edge-triggered: we fire once on the outside->inside
  transition, not on every frame the person is inside (that would flood
  the log with thousands of near-duplicate events for one visit).
- Loitering is level-triggered against a time threshold: we fire once
  the *continuous* dwell time in the zone crosses
  `loiter_threshold_seconds`. It does not re-fire every frame after
  that, only once per continuous dwell.

Occlusion / ID-switch tolerance
--------------------------------
Trackers occasionally drop a person for a frame or two (occlusion, motion
blur) and reassign a new ID on reappearance. Two different failure modes,
two different mitigations:
  1. Missing detection for a *few* frames but *same* ID (tracker's own
     Kalman-filter prediction bridges the gap) -- handled automatically,
     nothing to do here.
  2. Missing detection long enough that the tracker gives up and assigns
     a *new* ID on re-entry -- this looks like a fresh person to us, so
     dwell time restarts. This is a known limitation (see README); a
     production system would add appearance-based re-ID matching to
     stitch broken tracks back together.

We also grace-tolerate brief drop-outs *within* a zone: a track that
disappears for <= `MAX_MISS_FRAMES_FOR_ZONE_CONTINUITY` frames and then
reappears inside the same zone keeps its dwell timer instead of
resetting to zero, since single-frame misses inside a crowd are common
and shouldn't punish an otherwise continuously-loitering person.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector_tracker import Detection
from .zones import Zone

MAX_MISS_FRAMES_FOR_ZONE_CONTINUITY = 8  # ~0.3s at 25fps


@dataclass
class Event:
    frame: int
    timestamp: float
    track_id: int
    zone_name: str
    event_type: str  # "intrusion" | "loitering"
    bbox: tuple[int, int, int, int]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "timestamp_s": round(self.timestamp, 3),
            "track_id": self.track_id,
            "zone": self.zone_name,
            "event_type": self.event_type,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class _TrackZoneState:
    inside: bool = False
    entered_at: float | None = None
    intrusion_fired: bool = False
    loiter_fired: bool = False
    last_inside_frame: int = -1


class EventDetector:
    def __init__(self, zones: list[Zone]):
        self.zones = zones
        # state[(track_id, zone_name)] -> _TrackZoneState
        self.state: dict[tuple[int, str], _TrackZoneState] = {}
        self.active_tracks_current_frame: set[tuple[int, str]] = set()

    def update(self, frame_idx: int, timestamp: float, detections: list[Detection]) -> list[Event]:
        new_events: list[Event] = []
        self.active_tracks_current_frame = set()

        for det in detections:
            fx, fy = det.foot_point
            for zone in self.zones:
                key = (det.track_id, zone.name)
                is_inside_now = zone.contains(fx, fy)

                st = self.state.get(key)
                if st is None:
                    st = _TrackZoneState()
                    self.state[key] = st

                if is_inside_now:
                    self.active_tracks_current_frame.add(key)

                    was_inside_recently = st.inside or (
                        frame_idx - st.last_inside_frame <= MAX_MISS_FRAMES_FOR_ZONE_CONTINUITY
                        and st.entered_at is not None
                    )

                    if not was_inside_recently:
                        # Fresh entry: reset dwell tracking.
                        st.entered_at = timestamp
                        st.intrusion_fired = False
                        st.loiter_fired = False

                    st.inside = True
                    st.last_inside_frame = frame_idx

                    # --- Intrusion: fire once on entry ---
                    if zone.type == "intrusion" and not st.intrusion_fired:
                        st.intrusion_fired = True
                        new_events.append(
                            Event(
                                frame=frame_idx,
                                timestamp=timestamp,
                                track_id=det.track_id,
                                zone_name=zone.name,
                                event_type="intrusion",
                                bbox=det.bbox,
                                confidence=det.conf,
                            )
                        )

                    # --- Loitering: fire once dwell exceeds threshold ---
                    if zone.type == "loitering" and not st.loiter_fired:
                        entered_at = st.entered_at if st.entered_at is not None else timestamp
                        dwell = timestamp - entered_at
                        if dwell >= zone.loiter_threshold_seconds:
                            st.loiter_fired = True
                            new_events.append(
                                Event(
                                    frame=frame_idx,
                                    timestamp=timestamp,
                                    track_id=det.track_id,
                                    zone_name=zone.name,
                                    event_type="loitering",
                                    bbox=det.bbox,
                                    confidence=det.conf,
                                )
                            )
                else:
                    # Not inside this frame. Only fully reset if the miss
                    # has exceeded our grace window (see module docstring).
                    if st.inside and frame_idx - st.last_inside_frame > MAX_MISS_FRAMES_FOR_ZONE_CONTINUITY:
                        st.inside = False
                        st.entered_at = None
                        st.intrusion_fired = False
                        st.loiter_fired = False

        self._prune_stale(frame_idx)
        return new_events

    def _prune_stale(self, frame_idx: int, max_age_frames: int = 300):
        """Drop bookkeeping for tracks not seen in a long time so memory
        doesn't grow unbounded on long videos (production concern)."""
        stale = [
            k for k, st in self.state.items()
            if frame_idx - st.last_inside_frame > max_age_frames and not st.inside
        ]
        for k in stale:
            del self.state[k]

    def current_dwell_info(self) -> dict[tuple[int, str], float]:
        """For visualization: how long each currently-inside track has
        been dwelling, keyed by (track_id, zone_name)."""
        out = {}
        for key in self.active_tracks_current_frame:
            st = self.state[key]
            out[key] = st.entered_at
        return out
