"""Generates a small synthetic test clip: two moving 'person-shaped'
blobs on a plain background, one of which walks through the intrusion
zone and one of which parks itself in the loitering zone. This is only
for exercising the pipeline plumbing (I/O, zone math, event state,
video writing) end-to-end without needing a real dataset download --
it is NOT a substitute for testing detection accuracy on the real
VIRAT/MOT17/UCF-Crime clips called for in the assignment.
"""
import cv2
import numpy as np

W, H, FPS, SECONDS = 720, 480, 25, 12
N = FPS * SECONDS

out = cv2.VideoWriter("tests/synthetic.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

for i in range(N):
    frame = np.full((H, W, 3), 40, dtype=np.uint8)

    # Walker: crosses through the intrusion zone (x:100-350, y:100-320)
    wx = int(50 + (i / N) * 500)
    wy = 200
    cv2.rectangle(frame, (wx - 15, wy - 40), (wx + 15, wy + 40), (200, 200, 200), -1)
    cv2.circle(frame, (wx, wy - 55), 12, (200, 200, 200), -1)

    # Loiterer: sits still inside the loitering zone (x:400-680, y:150-420) after frame 50
    if i > 40:
        lx, ly = 540, 300
        cv2.rectangle(frame, (lx - 15, ly - 40), (lx + 15, ly + 40), (180, 220, 180), -1)
        cv2.circle(frame, (lx, ly - 55), 12, (180, 220, 180), -1)

    out.write(frame)

out.release()
print(f"Wrote tests/synthetic.mp4 ({N} frames, {W}x{H} @ {FPS}fps)")
