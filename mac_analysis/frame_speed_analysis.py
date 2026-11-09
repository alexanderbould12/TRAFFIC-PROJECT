import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

BASE = Path.home() / "Documents/traffic_ml"
RESULTS = BASE / "results"

NAME = "traffic_2026-11-08_13-40-36"

DETECTIONS = RESULTS / f"{NAME}_detections.csv"
EVENTS = RESULTS / f"{NAME}_events.csv"
OUTPUT = RESULTS / f"{NAME}_frame_timing.csv"

FPS = 20.0
LINE_A = 140
LINE_B = 500
MAX_FRAME_GAP = 3

tracks = defaultdict(list)

with DETECTIONS.open(newline="") as f:
    for r in csv.DictReader(f):
        tracks[int(r["track_id"])].append({
            "frame": int(r["frame"]),
            "x": float(r["center_x"])
        })

with EVENTS.open(newline="") as f:
    events = list(csv.DictReader(f))

def crossing_frame(rows, line, direction):
    for before, after in zip(rows, rows[1:]):
        f0, f1 = before["frame"], after["frame"]
        x0, x1 = before["x"], after["x"]

        if f1 - f0 > MAX_FRAME_GAP:
            continue

        if direction == "left_to_right":
            crossed = x0 <= line <= x1 and x1 > x0
        else:
            crossed = x1 <= line <= x0 and x1 < x0

        if crossed:
            fraction = (line - x0) / (x1 - x0)
            return f0 + fraction * (f1 - f0)

    return None

results = []

for event in events:
    tid = int(event["track_id"])
    direction = event["direction"]
    rows = sorted(tracks[tid], key=lambda r: r["frame"])

    first_line, second_line = (
        (LINE_A, LINE_B)
        if direction == "left_to_right"
        else (LINE_B, LINE_A)
    )

    frame_a = crossing_frame(rows, first_line, direction)
    frame_b = crossing_frame(rows, second_line, direction)

    elapsed_frames = ""
    elapsed_seconds = ""

    if frame_a is not None and frame_b is not None:
        delta = frame_b - frame_a

        if delta > 0:
            elapsed_frames = round(delta, 2)
            elapsed_seconds = round(delta / FPS, 3)

    results.append({
        "track_id": tid,
        "vehicle_type": event["vehicle_type"],
        "direction": direction,
        "first_crossing_frame": (
            round(frame_a, 2) if frame_a is not None else ""
        ),
        "second_crossing_frame": (
            round(frame_b, 2) if frame_b is not None else ""
        ),
        "elapsed_frames": elapsed_frames,
        "elapsed_seconds": elapsed_seconds
    })

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("=== FRAME-BASED TIMING RESULTS ===")
print(f"Source FPS: {FPS}")
print(f"Measurement lines: x={LINE_A}, x={LINE_B}")
print(f"Candidate crossings: {len(results)}")

for direction in ["left_to_right", "right_to_left"]:
    times = [
        float(r["elapsed_seconds"])
        for r in results
        if r["direction"] == direction
        and r["elapsed_seconds"] != ""
    ]

    print(f"\n{direction}")
    print(f"Vehicles timed: {len(times)}")

    if times:
        print(f"Median travel time: {median(times):.3f} seconds")
        print(f"Fastest crossing: {min(times):.3f} seconds")
        print(f"Slowest crossing: {max(times):.3f} seconds")

print(f"\nSaved: {OUTPUT.resolve()}")
