import cv2
import json
import math
from pathlib import Path

IMAGE = Path("speed_calibration.jpg")
OUTPUT = Path("speed_reference.json")

image = cv2.imread(str(IMAGE))

if image is None:
    raise FileNotFoundError(IMAGE)

points = []

def redraw():
    display = image.copy()

    for i, (x, y) in enumerate(points):
        label = "A" if i == 0 else "B"

        cv2.circle(display, (x, y), 6, (0, 255, 0), -1)

        cv2.putText(
            display, f"{label} ({x}, {y})",
            (min(x + 10, display.shape[1] - 130),
             max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )

    if len(points) == 2:
        cv2.line(
            display, points[0], points[1],
            (0, 255, 255), 2
        )

        distance = math.dist(points[0], points[1])

        cv2.putText(
            display, f"Image separation: {distance:.1f} px",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

    cv2.imshow("Speed Calibration", display)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            redraw()

cv2.namedWindow("Speed Calibration")
cv2.setMouseCallback("Speed Calibration", on_click)

redraw()

print("Click two reference points.")
print("R = Reset | S = Save | Q = Quit")

while True:
    key = cv2.waitKey(30) & 0xFF

    if key == ord("r"):
        points.clear()
        print("Points reset.")
        redraw()

    elif key == ord("s"):
        if len(points) != 2:
            print("Select two points first.")
            continue

        distance_m = float(
            input("Real-world distance in meters: ")
        )

        data = {
            "point_a": points[0],
            "point_b": points[1],
            "distance_meters": distance_m,
            "calibration_status": "approximate"
        }

        OUTPUT.write_text(json.dumps(data, indent=2))

        print(f"Saved: {OUTPUT.resolve()}")
        break

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
