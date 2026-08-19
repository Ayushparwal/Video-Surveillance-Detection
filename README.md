# Video Surveillance: Detection, Tracking & Event Recognition

A pipeline that detects people in security footage, tracks them across frames with
persistent IDs, and raises **zone intrusion** and **loitering** events based on a
configurable JSON zone map.

## Architecture Overview

```
video file
   │
   ▼
┌─────────────────────┐
│  PersonTracker       │  YOLOv8 (person class) + ByteTrack
│  (detector_tracker.py)│  → list[Detection(track_id, bbox, conf)] per frame
└─────────┬────────────┘
          │
          ▼
┌─────────────────────┐      ┌──────────────────┐
│  EventDetector        │◄────│  Zone (zones.py)  │  polygon config, point-in-polygon
│  (events.py)           │      └──────────────────┘
│  per-(track,zone) state machine: intrusion (edge-triggered),
│  loitering (dwell-time level-triggered)
└─────────┬────────────┘
          │  new Event objects
          ▼
┌─────────────────────┐     ┌────────────────────┐
│  visualizer.py        │     │  EventLogger        │
│  draws boxes/IDs/zones│     │  (logger.py)         │
│  → annotated frame     │     │  → events.json/.csv │
└─────────┬────────────┘     └────────────────────┘
          │
          ▼
   annotated .mp4
```

Each stage is a separate module with a narrow interface (`Detection` in, `Detection`
out; `Event` objects as the only channel between event logic and everything downstream).
This means:
- The detector/tracker can be swapped (e.g. `yolov8n.pt` → a fine-tuned model, or
  ByteTrack → BoT-SORT) without touching zone or event code.
- Event logic is unit-testable with plain `Detection` objects and no video/model
  dependency at all (see `tests/test_events.py`).
- `pipeline.py` only does orchestration and frame I/O — no business logic lives there.

## Model Choices

### Detector: YOLOv8n (nano)
**Alternatives considered:** Faster R-CNN, YOLOv8s/m, RT-DETR.

- **YOLO family over Faster R-CNN**: Faster R-CNN's two-stage region-proposal design
  gets marginally higher mAP on small/occluded objects, but is 3-5x slower per frame
  on CPU and has no first-class multi-object tracking integration. For a surveillance
  pipeline that needs to process hours of footage, single-stage YOLO's speed/accuracy
  trade-off wins, and it's the de facto standard for this exact use case (VIRAT/MOT
  benchmarks are dominated by YOLO submissions).
- **`yolov8n` over `yolov8s/m`**: nano is the right default for a CPU-only prototype —
  it hits real-time-ish speeds without a GPU. `--model yolov8s.pt` is a one-flag swap
  if higher accuracy is worth the ~3x slowdown (e.g. dense crowds, small/far-away
  people, or when a GPU is available). This is exposed as a CLI flag specifically so
  reviewers can trade off speed vs. accuracy without touching code.
- **Not fine-tuned**: COCO's `person` class is broad enough to work directly on
  VIRAT/MOT17/UCF-Crime without domain-specific fine-tuning, which matches the
  assignment's "use existing models, show engineering judgment" framing. A production
  system would fine-tune on the target camera's actual footage (different angles,
  resolutions, and lighting than COCO's photos) — noted under Limitations.

### Tracker: ByteTrack (via Ultralytics `.track()`)
**Alternatives considered:** DeepSORT, StrongSORT, BoT-SORT.

- **ByteTrack over DeepSORT**: DeepSORT's main advantage is an appearance embedding
  (a small Re-ID CNN) that helps re-match tracks after occlusion. ByteTrack instead
  keeps *all* detections — even low-confidence ones the naive approach would discard —
  and associates them with existing tracks via IoU + Kalman-filter motion prediction.
  In practice this makes ByteTrack more robust to partial occlusion (a person's box
  shrinking/getting a lower score) without the extra latency and model weights of a
  Re-ID network, and it's what Ultralytics ships as the default. For short-to-medium
  gaps this matches DeepSORT's practical accuracy at a fraction of the compute.
- **BoT-SORT** (Ultralytics' other built-in option) adds camera-motion compensation and
  an optional Re-ID model on top of ByteTrack — better for panning/moving cameras, at
  extra cost. It's a one-line config swap (`tracker='botsort.yaml'`) if the target
  footage has camera shake or pans; noted in Limitations since ByteTrack was chosen as
  the faster default for mostly-static CCTV framing.
- **Trade-off accepted**: neither ByteTrack nor BoT-SORT does full appearance-based
  Re-ID by default, so a person who fully leaves the frame for an extended period and
  re-enters will get a **new** track ID rather than resuming their old one. This is the
  single biggest limitation of the tracking layer — see Known Limitations.

## Setup

```bash
git clone Ayushparwal/Video-Surveillance-Detection
cd surveillance_system
pip install -r requirements.txt
```

Or with Docker:
```bash
docker build -t surveillance .
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results surveillance \
    --video data/input.mp4 --zones config/zones.json --output results/
```

YOLOv8 weights download automatically on first run (~6MB for `yolov8n.pt`).

## Running

```bash
python run.py --video input.mp4 --zones config/zones.json --output results/
```

Options:
| Flag | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | any Ultralytics YOLOv8 weights path/name |
| `--conf` | `0.35` | detection confidence threshold |
| `--device` | auto | `cpu`, `cuda`, `mps` |
| `--frame-stride` | `1` | process every Nth frame (speed vs. temporal resolution) |
| `--no-video-output` | off | skip annotated video, only emit event logs (faster) |
| `--verbose` | off | debug-level logging |

## Configuring Zones

`config/zones.json`:
```json
{
  "zones": [
    {
      "name": "restricted_entrance",
      "type": "intrusion",
      "polygon": [[100, 100], [350, 100], [350, 320], [100, 320]],
      "color": [0, 0, 255]
    },
    {
      "name": "lobby_waiting_area",
      "type": "loitering",
      "polygon": [[400, 150], [680, 150], [680, 420], [400, 420]],
      "loiter_threshold_seconds": 8,
      "color": [0, 165, 255]
    }
  ]
}
```
- `polygon` points are in **pixel coordinates** of the source video, ordered around
  the boundary (not necessarily convex — any simple polygon works).
- `type: "intrusion"` fires one event per continuous visit, on entry.
- `type: "loitering"` fires one event per continuous visit, once dwell time crosses
  `loiter_threshold_seconds`.
- The same physical area can appear as two zone entries (one of each type) if you want
  both "alert immediately on entry" and "escalate if they don't leave."
- Zone membership is tested against a detection's **foot point** (bottom-center of the
  bounding box), not the box centroid — this matches where the person is actually
  standing on the ground plane.

## Event Log Format

`results/events.json`:
```json
{
  "source_video": "input.mp4",
  "meta": { "processed_frames": 1500, "avg_processing_fps": 11.2, ... },
  "event_count": 3,
  "events": [
    {
      "frame": 412,
      "timestamp_s": 16.48,
      "track_id": 7,
      "zone": "restricted_entrance",
      "event_type": "intrusion",
      "bbox": [210, 140, 265, 310],
      "confidence": 0.87
    }
  ]
}
```
`events.csv` mirrors the same data in flat form for spreadsheet review.

## Sample Results

`sample_output/zone_overlay_demo.png` — a frame from a smoke-test run showing zone
overlays (red = intrusion zone, orange = loitering zone), confirming the rendering
and geometry pipeline. **This was generated on a synthetic placeholder clip**
(`tests/make_synthetic_video.py`) for CI/plumbing verification only, since this
environment has no internet access to the VIRAT/MOT17/UCF-Crime dataset hosts. Run
against a real clip — e.g. a MOT17 sequence — for actual detection/tracking/event
output; the pipeline itself is dataset-agnostic and takes any mp4/avi input.

To reproduce the smoke test:
```bash
python tests/make_synthetic_video.py
python run.py --video tests/synthetic.mp4 --zones config/zones.json --output tests/smoke_output
```

## Known Limitations

- **No long-gap re-identification.** If a person leaves the frame entirely for more
  than a couple seconds and re-enters, ByteTrack assigns a new track ID — the system
  sees them as a different person and any dwell-time state resets. Fix: add an
  appearance-embedding Re-ID model (e.g. OSNet) and match new tracks against a
  short-term gallery of recently-lost tracks by embedding distance, not just IoU/motion.
- **Static camera assumption.** Zone polygons are defined in fixed pixel coordinates.
  A panning/zooming camera invalidates them. Fix: either lock to static cameras (as
  most CCTV is) or add homography-based zone re-projection / switch to BoT-SORT's
  camera-motion compensation.
- **Occlusion in dense crowds.** Heavy overlap between people confuses the Kalman-filter
  motion model and increases ID switches; box-only IoU tracking has no way to
  disambiguate two people crossing paths in a tight crowd. A Re-ID embedding (see
  above) would also help here.
- **Loitering ≠ stationary, currently.** The current definition is "continuous dwell
  time inside the zone," which counts a person pacing back and forth inside a zone as
  loitering. A stricter definition (checking bounding-box centroid displacement stays
  below a pixel threshold, not just zone membership) would be a small addition to
  `events.py` if the use case needs "truly motionless" rather than "present."
- **No dataset-specific evaluation script.** Given the time budget, I prioritized the
  core pipeline over a MOTA/MOTP evaluation harness against MOT17 ground truth — noted
  as the first stretch item if extending this.
- **Confidence threshold is global**, not per-zone or per-lighting-condition. A single
  clip spanning both bright outdoor and dim indoor areas may need different thresholds
  per region; not currently supported.

## Performance Notes

Measured on this development environment (CPU-only, no GPU, `yolov8n.pt`,
720×480 synthetic clip, single-threaded):
- ~8 FPS end-to-end (detection + tracking + zone logic + annotated-video encoding)
- Memory stays flat across the run — event state is pruned for tracks unseen for
  300+ frames (`EventDetector._prune_stale`), so long videos don't leak memory via
  ever-growing per-track dictionaries.
- `--no-video-output` skips frame encoding/writing, which is the single biggest cost
  after model inference — useful for a first pass over long footage where only the
  event log is needed.
- `--frame-stride N` linearly reduces both compute and (proportionally) tracking
  temporal resolution — reasonable for N=2-3 on stationary/slow scenes, risky for
  fast motion since ByteTrack's association degrades with larger inter-frame jumps.
- On a GPU (`--device cuda`), `yolov8n` typically reaches 60-100+ FPS at this
  resolution — CPU numbers above are a conservative lower bound.

## Production Thinking (beyond this prototype's scope)

- **Deduplication/alerting**: current dedup is per continuous dwell (one event per
  visit). A real alerting layer would add cooldown windows per (camera, zone) to avoid
  paging someone every time a queue re-forms, plus severity levels and an
  acknowledge/suppress workflow.
- **Multi-camera**: the pipeline is single-video by design; extending to multi-camera
  is mostly a wrapper that runs N pipeline instances and merges event streams by
  timestamp — no changes needed to the core detection/event logic.
- **GPU/CPU awareness**: `--device` is exposed explicitly rather than silently
  auto-detecting, so deployment configs are reproducible across machines.
- **Reproducibility**: pinned `requirements.txt`, a `Dockerfile`, and a fixed
  `tracker='bytetrack.yaml'` config path (rather than relying on library defaults that
  could change between ultralytics versions).
