import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path.home() / "Documents/traffic_ml"
MASTER = BASE / "master_traffic_data.csv"
RESULTS = BASE / "results"

LINE_RATIO = 0.5
MARGIN = 40
MIN_FRAMES = 5
MIN_MOVEMENT = 120
MAX_CROSSING_GAP = 12

COLUMNS = [
    "recording", "recording_start", "video_time_seconds",
    "track_id", "vehicle_type", "direction", "speed_mph",
    "validation_status"
]

parser = argparse.ArgumentParser()
parser.add_argument("video", help="Path to an MP4 recording")
args = parser.parse_args()

video = Path(args.video).expanduser().resolve()
if not video.is_file():
    raise FileNotFoundError(video)

RESULTS.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError("Could not open recording")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

if fps <= 0 or width <= 0:
    raise RuntimeError("Invalid recording metadata")

line_x = width * LINE_RATIO
model = YOLO(str(BASE / "yolo11n.pt"))

observations = []
tracks = defaultdict(list)
frame_number = 0

print(f"Processing: {video.name}", flush=True)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_number += 1

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[1, 2, 3, 5, 7],
            conf=0.20,
            imgsz=640,
            verbose=False
        )[0]

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        for box in boxes:
            track_id = int(box.id.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            row = {
                "frame": frame_number,
                "video_time_seconds": round(
                    (frame_number - 1) / fps, 3
                ),
                "track_id": track_id,
                "vehicle_type": model.names[int(box.cls.item())],
                "confidence": round(float(box.conf.item()), 3),
                "center_x": round((x1 + x2) / 2, 2),
                "center_y": round((y1 + y2) / 2, 2),
                "box_width": round(x2 - x1, 2),
                "box_height": round(y2 - y1, 2)
            }

            observations.append(row)
            tracks[track_id].append(row)

finally:
    cap.release()

# Calculate crossing events
events = []

for track_id, rows in tracks.items():
    rows.sort(key=lambda r: r["frame"])

    if len(rows) < MIN_FRAMES:
        continue

    xs = [r["center_x"] for r in rows]

    if max(xs) - min(xs) < MIN_MOVEMENT:
        continue

    vehicle_type = Counter(
        r["vehicle_type"] for r in rows
    ).most_common(1)[0][0]

    previous = None

    for row in rows:
        x = row["center_x"]

        if x < line_x - MARGIN:
            side = "left"
        elif x > line_x + MARGIN:
            side = "right"
        else:
            continue

        if previous is not None:
            old_side, old_row = previous

            if old_side != side:
                frame_gap = row["frame"] - old_row["frame"]

                if frame_gap <= MAX_CROSSING_GAP:
                    direction = (
                        "left_to_right"
                        if old_side == "left"
                        else "right_to_left"
                    )

                    events.append({
                        "recording": video.name,
                        "recording_start": "",
                        "video_time_seconds":
                            row["video_time_seconds"],
                        "track_id": track_id,
                        "vehicle_type": vehicle_type,
                        "direction": direction,
                        "speed_mph": "",
                        "validation_status": "candidate"
                    })

                    break

        previous = (side, row)

events.sort(key=lambda e: float(e["video_time_seconds"]))

# Save individual processing results
detection_file = RESULTS / f"{video.stem}_detections.csv"
event_file = RESULTS / f"{video.stem}_events.csv"

with detection_file.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(observations[0].keys())
        if observations else [
            "frame", "video_time_seconds", "track_id",
            "vehicle_type", "confidence", "center_x",
            "center_y", "box_width", "box_height"
        ]
    )
    writer.writeheader()
    writer.writerows(observations)

with event_file.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(events)

# Update master, without duplicating previous recordings
existing = []

if MASTER.exists():
    with MASTER.open(newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != COLUMNS:
            raise ValueError("Master CSV has unexpected columns")

        existing = list(reader)

already_imported = any(
    row["recording"] == video.name
    for row in existing
)

if already_imported:
    added = 0
    print("Recording already in master CSV; no duplicates added.")
else:
    existing.extend(events)
    added = len(events)

    with MASTER.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(existing)

print("\n=== PROCESSING SUMMARY ===")
print(f"Frames processed: {frame_number}")
print(f"Detection observations: {len(observations)}")
print(f"Unique tracking IDs: {len(tracks)}")
print(f"Candidate crossings: {len(events)}")
print(f"New master events: {added}")
print(f"Total master events: {len(existing)}")
print(f"Results folder: {RESULTS}")
