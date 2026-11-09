import cv2
import csv
from pathlib import Path
from collections import defaultdict

VIDEO = Path("recordings/daytime_test_2.mp4")
DETECTIONS = Path("vehicle_detections.csv")
EVENTS = Path("traffic_events.csv")
OUTPUT = Path("traffic_counting_visualized.mp4")

LINE_X = 640

# Load tracked vehicle positions
frame_detections = defaultdict(list)

with DETECTIONS.open(newline="") as f:
    for row in csv.DictReader(f):
        frame_detections[int(row["frame"])].append(row)

# Load counting events
events = []

with EVENTS.open(newline="") as f:
    for row in csv.DictReader(f):
        events.append(row)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Invalid video FPS")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video")

# Convert event times to frame numbers
events_by_frame = defaultdict(list)

for event in events:
    event_frame = round(
        float(event["video_time_seconds"]) * fps
    ) + 1

    events_by_frame[event_frame].append(event)

frame_number = 0
left_to_right = 0
right_to_left = 0
counted_ids = set()

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Virtual counting line
        cv2.line(
            frame,
            (LINE_X, 0),
            (LINE_X, height),
            (0, 255, 255),
            3
        )

        # Update crossing counts
        for event in events_by_frame.get(frame_number, []):
            track_id = int(event["track_id"])
            counted_ids.add(track_id)

            if event["direction"] == "left_to_right":
                left_to_right += 1
            else:
                right_to_left += 1

        # Draw tracked vehicle boxes
        for row in frame_detections.get(frame_number, []):
            track_id = int(row["track_id"])

            cx = float(row["center_x"])
            cy = float(row["center_y"])

            w = float(row["box_width"])
            h = float(row["box_height"])

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # Green indicates a track that has crossed
            color = (
                (0, 255, 0)
                if track_id in counted_ids
                else (255, 120, 0)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = (
                f"ID {track_id} "
                f"{row['vehicle_type']}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # Dark background for count display
        cv2.rectangle(
            frame,
            (0, 0),
            (410, 110),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"LEFT -> RIGHT: {left_to_right}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"RIGHT -> LEFT: {right_to_left}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"TOTAL: {left_to_right + right_to_left}",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        writer.write(frame)

finally:
    cap.release()
    writer.release()

print("=== TRAFFIC COUNTING RESULTS ===")
print(f"Frames processed: {frame_number}")
print(f"Left to right: {left_to_right}")
print(f"Right to left: {right_to_left}")
print(f"Total crossings: {left_to_right + right_to_left}")
print(f"Saved video: {OUTPUT.resolve()}")
