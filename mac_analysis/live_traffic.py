import csv
import subprocess
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path.home() / "Documents/traffic_ml"
MASTER = BASE / "master_traffic_data.csv"

STREAM = "tcp://192.168.7.60:8888"
WIDTH, HEIGHT = 640, 360
FPS = 15
LINE_X = WIDTH // 2
MARGIN = 20

COLUMNS = [
    "recording", "recording_start", "video_time_seconds",
    "track_id", "vehicle_type", "direction",
    "speed_mph", "validation_status"
]

session_start = datetime.now().astimezone()
session_id = "live_" + session_start.strftime("%Y%m%d_%H%M%S")

if MASTER.exists():
    with MASTER.open(newline="") as f:
        if csv.DictReader(f).fieldnames != COLUMNS:
            raise RuntimeError("Unexpected master CSV columns")
else:
    with MASTER.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

model = YOLO(str(BASE / "yolo11n.pt"))

command = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-fflags", "nobuffer", "-f", "h264",
    "-i", STREAM,
    "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}",
    "-pix_fmt", "bgr24",
    "-f", "rawvideo", "pipe:1"
]

process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)

frame_size = WIDTH * HEIGHT * 3

# Continuously drain FFmpeg so old frames cannot build up.
condition = threading.Condition()
latest = {"frame": None, "seq": 0, "closed": False}

def camera_reader():
    while True:
        data = bytearray()

        while len(data) < frame_size:
            chunk = process.stdout.read(frame_size - len(data))
            if not chunk:
                break
            data.extend(chunk)

        if len(data) != frame_size:
            break

        frame = np.frombuffer(
            data, dtype=np.uint8
        ).reshape(HEIGHT, WIDTH, 3).copy()

        with condition:
            latest["frame"] = frame
            latest["seq"] += 1
            condition.notify()

    with condition:
        latest["closed"] = True
        condition.notify_all()

reader = threading.Thread(
    target=camera_reader, daemon=True
)
reader.start()

last_seq = 0
processed = 0
last_report = time.monotonic()

tracks = defaultdict(lambda: {
    "first_side": None,
    "last_side": None,
    "frames": 0,
    "min_x": float("inf"),
    "max_x": float("-inf"),
    "types": Counter(),
    "counted": False
})

total = 0

print("=== LIVE TRAFFIC MONITOR ===", flush=True)
print("Camera:", STREAM, flush=True)
print("CSV:", MASTER, flush=True)
print("Press Ctrl+C to stop.\n", flush=True)

try:
    while True:
        # Use the newest decoded frame, skipping stale frames.
        with condition:
            condition.wait_for(
                lambda: latest["seq"] > last_seq
                or latest["closed"],
                timeout=5
            )

            if latest["seq"] == last_seq:
                if latest["closed"]:
                    print("Camera disconnected.", flush=True)
                    break
                continue

            frame = latest["frame"]
            last_seq = latest["seq"]

        processed += 1
        now_perf = time.monotonic()

        if now_perf - last_report >= 10:
            rate = processed / (now_perf - last_report)
            print(
                f"[Performance] Analysis: {rate:.1f} FPS | "
                f"Latest frame: {last_seq}",
                flush=True
            )
            processed = 0
            last_report = now_perf

        result = model.track(
            frame, persist=True,
            tracker="bytetrack.yaml",
            classes=[1, 2, 3, 5, 7],
            conf=0.15, imgsz=640,
            verbose=False
        )[0]

        if result.boxes is None or result.boxes.id is None:
            continue

        for box in result.boxes:
            tid = int(box.id.item())
            vehicle_type = model.names[int(box.cls.item())]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2

            track = tracks[tid]
            track["frames"] += 1
            track["types"][vehicle_type] += 1
            track["min_x"] = min(track["min_x"], cx)
            track["max_x"] = max(track["max_x"], cx)

            if track["counted"]:
                continue

            if cx < LINE_X - MARGIN:
                side = "left"
            elif cx > LINE_X + MARGIN:
                side = "right"
            else:
                continue

            if track["first_side"] is None:
                track["first_side"] = side

            track["last_side"] = side

            movement = track["max_x"] - track["min_x"]

            if (
                track["first_side"] != track["last_side"]
                and track["frames"] >= 2
                and movement >= 35
            ):
                direction = (
                    "left_to_right"
                    if track["first_side"] == "left"
                    else "right_to_left"
                )

                now = datetime.now().astimezone()
                elapsed = (
                    now - session_start
                ).total_seconds()

                event = {
                    "recording": session_id,
                    "recording_start": session_start.isoformat(),
                    "video_time_seconds": round(elapsed, 2),
                    "track_id": tid,
                    "vehicle_type": track["types"].most_common(1)[0][0],
                    "direction": direction,
                    "speed_mph": "",
                    "validation_status": "candidate"
                }

                with MASTER.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS)
                    writer.writerow(event)
                    f.flush()

                track["counted"] = True
                total += 1

                print(
                    f"{now.strftime('%H:%M:%S')} | "
                    f"{vehicle_type} | {direction} | "
                    f"TOTAL: {total}",
                    flush=True
                )

except KeyboardInterrupt:
    print("\nStopping monitor.")

finally:
    process.terminate()
    process.wait()
    print(f"Total candidate crossings: {total}")
    print(f"Saved to: {MASTER}")
