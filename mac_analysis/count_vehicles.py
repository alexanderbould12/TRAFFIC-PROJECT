import csv
from collections import defaultdict
from pathlib import Path

INPUT = Path("vehicle_detections.csv")
OUTPUT = Path("traffic_events.csv")

LINE_X = 640
MARGIN = 40
MIN_FRAMES = 5
MIN_MOVEMENT = 120

tracks = defaultdict(list)

with INPUT.open(newline="") as f:
    for row in csv.DictReader(f):
        tracks[int(row["track_id"])].append(row)

events = []

for track_id, rows in tracks.items():
    rows.sort(key=lambda r: int(r["frame"]))

    if len(rows) < MIN_FRAMES:
        continue

    xs = [float(r["center_x"]) for r in rows]
    movement = max(xs) - min(xs)

    if movement < MIN_MOVEMENT:
        continue

    previous_side = None

    for row in rows:
        x = float(row["center_x"])

        if x < LINE_X - MARGIN:
            side = "left"
        elif x > LINE_X + MARGIN:
            side = "right"
        else:
            continue

        if previous_side is not None and previous_side != side:
            direction = (
                "left_to_right"
                if previous_side == "left"
                else "right_to_left"
            )

            events.append({
                "video_time_seconds": row["video_time_seconds"],
                "track_id": track_id,
                "vehicle_type": row["vehicle_type"],
                "direction": direction
            })

            break

        previous_side = side

events.sort(key=lambda e: float(e["video_time_seconds"]))

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video_time_seconds",
            "track_id",
            "vehicle_type",
            "direction"
        ]
    )
    writer.writeheader()
    writer.writerows(events)

print(f"Candidate vehicle crossings: {len(events)}")

for direction in ["left_to_right", "right_to_left"]:
    count = sum(e["direction"] == direction for e in events)
    print(f"{direction}: {count}")

print(f"Saved to: {OUTPUT.resolve()}")
