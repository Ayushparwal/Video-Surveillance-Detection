#!/usr/bin/env python3
"""
CLI entrypoint.

Usage:
    python run.py --video input.mp4 --zones zones.json --output results/

    # tweak model / confidence / device
    python run.py --video input.mp4 --zones zones.json --output results/ \\
        --model yolov8s.pt --conf 0.4 --device cpu

    # process every 2nd frame for speed on long videos
    python run.py --video input.mp4 --zones zones.json --output results/ --frame-stride 2
"""

from __future__ import annotations

import argparse
import sys

from src.pipeline import SurveillancePipeline


def parse_args():
    p = argparse.ArgumentParser(description="Video surveillance: detection, tracking & event recognition")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--zones", required=True, help="Path to zones.json config")
    p.add_argument("--output", required=True, help="Output directory for annotated video + event logs")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO weights (default: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold (default: 0.35)")
    p.add_argument("--device", default=None, help="'cpu', 'cuda', 'mps', or leave unset for auto")
    p.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame (default: 1 = every frame)")
    p.add_argument("--no-video-output", action="store_true", help="Skip writing annotated video, only emit event logs")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()



def main():
    args = parse_args()
    pipeline = SurveillancePipeline(
        video_path=args.video,
        zones_path=args.zones,
        output_dir=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        frame_stride=args.frame_stride,
        no_video_output=args.no_video_output,
        verbose=args.verbose,
    )
    try:
        result = pipeline.run()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone: {result['event_count']} events detected.")
    print(f"Event log (JSON): {result['events_path']}")
    print(f"Event log (CSV):  {result['csv_path']}")
    if result["video_path"]:
        print(f"Annotated video:  {result['video_path']}")


if __name__ == "__main__":
    main()
