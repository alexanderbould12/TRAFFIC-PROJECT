import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path.home() / "Documents/traffic_ml"
RESULTS = BASE / "results"

NAME = "traffic_2026-11-08_13-40-36"
DETECTIONS = RESULTS / f"{NAME}_detections.csv"
EVENTS = RESULTS / f"{NAME}_events.csv"
OUTPUT = RESULTS / f"{NAME}_speed_experiment.csv"

REFERENCE = BASE / "speed_reference.json"

# Use the selected calibration points.
with REFERENCE.open() as f:
    calibration = json.load(f)

ax = float(calibration["point_a"][0])
bx = float(calibration["point_b"][0])
distance_m = float(calibration["distance_meters"])

if ax == bx or distance_m <= 0:
    raise ValueError("Invalid calibration")

# Interior measurement lines, away from frame edges.
LINE_A = 140
LINE_B = 500

# Rough initial scale ONLY; perspective is not corrected.
meters_per_pixel = distance_m / abs(bx - ax)
test_distance_m = abs(LINE_B - LINE_A) * meters_per_pixel

tracks = defaultdict(list)

with DETECTIONS.open(newline="") as f:
    for row in csv.DictReader(f):
        tracks[int(row["track_id"])].append({
            "frame": int(row["frame"]),
            "time": float(row["video_time_seconds"]),
            "x": float(row["center_x"])
        })

with EVENTS.open(newline="") as f:
    events = list(csv.DictReader(f))


def crossing_time(rows, line, direction):
    """Interpolate when a tracked center crosses a line."""
    for before, after in zip(rows, rows[1:]):
        x0, x1 = before["x"], after["x"]

        # Reject crossings with large detection gaps.
        if after["frame"] - before["frame"] > 3:
            continue

        if direction == "left_to_right":
            crossed = x0 <= line <= x1 and x1 > x0
        else:
            crossed = x1 <= line <= x0 and x1 < x0

        if not crossed:
            continue

        fraction = (line - x0) / (x1 - x0)

        return (
            before["time"]
            + fraction * (after["time"] - before["time"])
        )

    return None


output_rows = []

for event in events:
    tid = int(event["track_id"])
    direction = event["direction"]
    rows = sorted(tracks[tid], key=lambda r: r["frame"])

    if direction == "left_to_right":
        first_line, second_line = LINE_A, LINE_B
    else:
        first_line, second_line = LINE_B, LINE_A

    t1 = crossing_time(rows, first_line, direction)
    t2 = crossing_time(rows, second_line, direction)

    speed_mph = ""

    if t1 is not None and t2 is not None:
        elapsed = t2 - t1

        if elapsed > 0:
            candidate_speed = (
                test_distance_m / elapsed * 2.236936
            )

            # Reject implausible estimates.
            if 3 <= candidate_speed <= 85:
                speed_mph = round(candidate_speed, 1)

    output_rows.append({
        "track_id": tid,
        "vehicle_type": event["vehicle_type"],
        "direction": direction,
        "time_seconds": event["video_time_seconds"],
        "speed_mph_experimental": speed_mph,
        "calibration": "uncorrected_perspective"
    })

OUTPUT.parent.mkdir(exist_ok=True)

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(output_rows[0].keys())
        if output_rows else [
            "track_id", "vehicle_type", "direction",
            "time_seconds", "speed_mph_experimental",
            "calibration"
        ]
    )
    writer.writeheader()
    writer.writerows(output_rows)

measured = [
    r for r in output_rows
    if r["speed_mph_experimental"] != ""
]

print("=== EXPERIMENTAL SPEED RESULTS ===")
print(f"Reference distance: {distance_m:.2f} meters")
print(f"Test zone: x={LINE_A} to x={LINE_B}")
print(f"Assumed test-zone distance: {test_distance_m:.2f} meters")
print(f"Candidate vehicles: {len(output_rows)}")
print(f"Vehicles with provisional speeds: {len(measured)}")

if measured:
    speeds = [
        float(r["speed_mph_experimental"])
        for r in measured
    ]
    print(f"Mean provisional speed: {sum(speeds)/len(speeds):.1f} MPH")

print(f"Saved: {OUTPUT.resolve()}")
