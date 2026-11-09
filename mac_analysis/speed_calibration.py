import cv2
from pathlib import Path

VIDEO = Path("recordings/traffic_2026-11-08_13-40-36.mp4")

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Cannot open the five-minute recording")

cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read video frame")

frame = cv2.resize(frame, (640, 360))

# Proposed speed measurement lines
LINE_A = 220
LINE_B = 420

for x, label in [(LINE_A, "A"), (LINE_B, "B")]:
    cv2.line(frame, (x, 0), (x, 360), (0, 255, 255), 2)
    cv2.putText(
        frame, f"Line {label}: x={x}",
        (x - 70, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (0, 255, 255), 2
    )

cv2.imwrite("speed_calibration.jpg", frame)

print("Created speed_calibration.jpg")
