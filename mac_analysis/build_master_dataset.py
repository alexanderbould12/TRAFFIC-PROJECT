import csv
from pathlib import Path

SOURCE = Path("traffic_events.csv")
MASTER = Path("master_traffic_data.csv")
RECORDING = "daytime_test_2.mp4"

COLUMNS = [
    "recording",
    "recording_start",
    "video_time_seconds",
    "track_id",
    "vehicle_type",
    "direction",
    "speed_mph",
    "validation_status"
]

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

# Preserve previously collected observations
existing = []

if MASTER.exists():
    with MASTER.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected master CSV columns")
        existing = list(reader)

# Prevent repeated imports of the same crossing
known = {
    (r["recording"], r["track_id"], r["video_time_seconds"])
    for r in existing
}

added = 0

with SOURCE.open(newline="") as f:
    for event in csv.DictReader(f):

        key = (
            RECORDING,
            event["track_id"],
            event["video_time_seconds"]
        )

        if key in known:
            continue

        existing.append({
            "recording": RECORDING,
            "recording_start": "",
            "video_time_seconds": event["video_time_seconds"],
            "track_id": event["track_id"],
            "vehicle_type": event["vehicle_type"],
            "direction": event["direction"],
            "speed_mph": "",
            "validation_status": "candidate"
        })

        known.add(key)
        added += 1

with MASTER.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(existing)

print("=== MASTER TRAFFIC DATASET ===")
print(f"New crossings added: {added}")
print(f"Total stored events: {len(existing)}")
print(f"Saved to: {MASTER.resolve()}")
