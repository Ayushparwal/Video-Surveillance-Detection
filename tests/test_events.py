"""Unit tests for the event state machine, independent of YOLO/tracking.
Run with: python -m pytest tests/test_events.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector_tracker import Detection
from src.events import EventDetector
from src.zones import load_zones

ZONES_PATH = Path(__file__).resolve().parent.parent / "config" / "zones.json"


def make_det(track_id, cx, cy, w=30, h=80, conf=0.9):
    return Detection(track_id=track_id, x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2, conf=conf)


def test_intrusion_fires_once_on_entry():
    zones = load_zones(ZONES_PATH)
    ed = EventDetector(zones)

    # Track 1 walks into the intrusion zone (100,100)-(350,320) and stays several frames.
    events_all = []
    for f in range(5):
        det = make_det(track_id=1, cx=200, cy=200)  # inside restricted_entrance
        events_all += ed.update(f, f / 25, [det])

    intrusion_events = [e for e in events_all if e.event_type == "intrusion"]
    assert len(intrusion_events) == 1, f"expected exactly 1 intrusion event, got {len(intrusion_events)}"
    assert intrusion_events[0].track_id == 1
    assert intrusion_events[0].zone_name == "restricted_entrance"


def test_intrusion_refires_after_exit_and_reentry():
    zones = load_zones(ZONES_PATH)
    ed = EventDetector(zones)

    events_all = []
    events_all += ed.update(0, 0.0, [make_det(1, 200, 200)])   # inside
    events_all += ed.update(1, 0.04, [make_det(1, 900, 900)])  # far outside any zone
    # grace window is 8 frames; push well past it
    for f in range(2, 15):
        events_all += ed.update(f, f / 25, [make_det(1, 900, 900)])
    events_all += ed.update(15, 0.6, [make_det(1, 200, 200)])  # back inside

    intrusion_events = [e for e in events_all if e.event_type == "intrusion"]
    assert len(intrusion_events) == 2, f"expected 2 intrusion events (exit+reentry), got {len(intrusion_events)}"


def test_loitering_fires_only_after_threshold():
    zones = load_zones(ZONES_PATH)
    ed = EventDetector(zones)
    loiter_zone = next(z for z in zones if z.type == "loitering")

    events_all = []
    fps = 25
    # loiter zone threshold is 8s -> below threshold, no event should fire yet
    for f in range(int(fps * 5)):
        det = make_det(track_id=2, cx=540, cy=300)  # inside lobby_waiting_area
        events_all += ed.update(f, f / fps, [det])
    assert not any(e.event_type == "loitering" for e in events_all), "loitering fired before threshold"

    # continue past threshold
    for f in range(int(fps * 5), int(fps * 10)):
        det = make_det(track_id=2, cx=540, cy=300)
        events_all += ed.update(f, f / fps, [det])

    loiter_events = [e for e in events_all if e.event_type == "loitering"]
    assert len(loiter_events) == 1, f"expected exactly 1 loitering event (deduped), got {len(loiter_events)}"


def test_person_outside_all_zones_generates_no_events():
    zones = load_zones(ZONES_PATH)
    ed = EventDetector(zones)
    events_all = []
    for f in range(50):
        det = make_det(track_id=3, cx=10, cy=10)  # top-left corner, outside both zones
        events_all += ed.update(f, f / 25, [det])
    assert events_all == []


def test_brief_occlusion_does_not_reset_loiter_timer():
    """A track that vanishes for a couple frames (occlusion) and reappears
    in the same zone should keep accumulating dwell time, not restart."""
    zones = load_zones(ZONES_PATH)
    ed = EventDetector(zones)
    fps = 25

    events_all = []
    for f in range(int(fps * 4)):
        events_all += ed.update(f, f / fps, [make_det(2, 540, 300)])
    # 3-frame occlusion (within the 8-frame grace window) -- no detection at all
    for f in range(int(fps * 4), int(fps * 4) + 3):
        events_all += ed.update(f, f / fps, [])
    # reappear, continue to threshold
    f_start = int(fps * 4) + 3
    for f in range(f_start, f_start + int(fps * 5)):
        events_all += ed.update(f, f / fps, [make_det(2, 540, 300)])

    loiter_events = [e for e in events_all if e.event_type == "loitering"]
    assert len(loiter_events) == 1, "dwell timer should survive brief occlusion"


if __name__ == "__main__":
    test_intrusion_fires_once_on_entry()
    test_intrusion_refires_after_exit_and_reentry()
    test_loitering_fires_only_after_threshold()
    test_person_outside_all_zones_generates_no_events()
    test_brief_occlusion_does_not_reset_loiter_timer()
    print("All tests passed.")
