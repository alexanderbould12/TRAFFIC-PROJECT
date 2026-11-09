import csv
import cv2
from pathlib import Path
from ultralytics import YOLO

VIDEO = Path("recordings/daytime_test_2.mp4")
OUTPUT = Path("vehicle_detections.csv")

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError("Cannot open daytime recording")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise RuntimeError("Could not determine video frame rate")

frame_number = 0
detections = 0
unique_ids = set()

with OUTPUT.open("w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "video_time_seconds",
        "frame",
        "track_id",
        "vehicle_type",
        "confidence",
        "center_x",
        "center_y",
        "box_width",
        "box_height"
    ])

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[1, 2, 3, 5, 7],
            conf=0.20,
            imgsz=640,
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            continue

        for box in boxes:
            track_id = int(box.id.item())
            vehicle_type = model.names[int(box.cls.item())]
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            writer.writerow([
                round((frame_number - 1) / fps, 3),
                frame_number,
                track_id,
                vehicle_type,
                round(confidence, 3),
                round((x1 + x2) / 2, 1),
                round((y1 + y2) / 2, 1),
                round(x2 - x1, 1),
                round(y2 - y1, 1)
            ])

            detections += 1
            unique_ids.add(track_id)

cap.release()

print(f"Frames processed: {frame_number}")
print(f"Detection observations: {detections}")
print(f"Unique tracking IDs: {len(unique_ids)}")
print(f"CSV saved to: {OUTPUT.resolve()}")
