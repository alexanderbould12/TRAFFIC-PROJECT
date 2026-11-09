import csv
import cv2
from collections import defaultdict
from pathlib import Path

VIDEO = Path("recordings/daytime_test_2.mp4")
OUTPUT = Path("crossing_verification.mp4")

with open("traffic_events.csv", newline="") as f:
    events = list(csv.DictReader(f))

tracks = defaultdict(dict)

with open("vehicle_detections.csv", newline="") as f:
    for row in csv.DictReader(f):
        tracks[int(row["track_id"])][int(row["frame"])] = row

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened() or fps <= 0:
    raise RuntimeError("Could not open video")

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video")

for i, event in enumerate(events, 1):
    tid = int(event["track_id"])
    timestamp = float(event["video_time_seconds"])

    start_frame = max(0, int((timestamp - 0.75) * fps))
    end_frame = int((timestamp + 0.75) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break

        frame_number = frame_idx + 1

        cv2.line(
            frame, (640, 0), (640, height),
            (0, 255, 255), 3
        )

        row = tracks[tid].get(frame_number)

        if row:
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            w = float(row["box_width"])
            h = float(row["box_height"])

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 4
            )

            cv2.circle(
                frame, (int(cx), int(cy)),
                7, (0, 0, 255), -1
            )

        cv2.rectangle(
            frame, (0, 0), (850, 65),
            (0, 0, 0), -1
        )

        text = (
            f"Event {i}/12 | ID {tid} | "
            f"{event['vehicle_type']} | "
            f"{event['direction']}"
        )

        cv2.putText(
            frame, text, (15, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        writer.write(frame)

cap.release()
writer.release()

print(f"Saved: {OUTPUT.resolve()}")
