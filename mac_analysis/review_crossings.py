import csv
import cv2
import numpy as np

with open("traffic_events.csv", newline="") as f:
    events = list(csv.DictReader(f))

cap = cv2.VideoCapture("recordings/daytime_test_2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

tiles = []

for i, event in enumerate(events, 1):
    seconds = float(event["video_time_seconds"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
    ok, frame = cap.read()

    if not ok:
        continue

    cv2.line(frame, (640, 0), (640, 720), (0, 255, 255), 4)

    frame = cv2.resize(frame, (640, 360))

    cv2.rectangle(frame, (0, 0), (640, 48), (0, 0, 0), -1)

    label = (
        f"Event {i} | {seconds:.1f}s | "
        f"ID {event['track_id']} | "
        f"{event['direction']}"
    )

    cv2.putText(
        frame, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2
    )

    tiles.append(frame)

cap.release()

if not tiles:
    raise RuntimeError("No crossing frames found")

blank = np.zeros_like(tiles[0])

while len(tiles) % 3:
    tiles.append(blank.copy())

rows = [
    np.hstack(tiles[i:i+3])
    for i in range(0, len(tiles), 3)
]

sheet = np.vstack(rows)
cv2.imwrite("crossing_review.jpg", sheet)

print(f"Created review sheet for {len(events)} events")
print("Saved: crossing_review.jpg")
