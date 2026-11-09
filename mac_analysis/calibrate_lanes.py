import cv2
import json
from pathlib import Path

IMAGE = Path("speed_calibration.jpg")
OUTPUT = Path("lane_calibration.json")

image = cv2.imread(str(IMAGE))

if image is None:
    raise FileNotFoundError(IMAGE)

LINE_A = 140
LINE_B = 500

points = []
names = ["A1 - Near", "B1 - Near", "A2 - Far", "B2 - Far"]
colors = [(0, 255, 0), (0, 255, 0),
          (255, 180, 0), (255, 180, 0)]

def redraw():
    display = image.copy()
    height = display.shape[0]

    for x in [LINE_A, LINE_B]:
        cv2.line(
            display, (x, 0), (x, height),
            (0, 255, 255), 2
        )

    for i, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 6, colors[i], -1)

        cv2.putText(
            display, names[i],
            (max(5, x - 60), max(18, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, colors[i], 2
        )

    if len(points) >= 2:
        cv2.line(
            display, points[0], points[1],
            (0, 255, 0), 2
        )

    if len(points) == 4:
        cv2.line(
            display, points[2], points[3],
            (255, 180, 0), 2
        )

    cv2.imshow("Lane Calibration", display)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 4:
            return

        # Snap clicks to the measurement lines
        line_x = (
            LINE_A if len(points) % 2 == 0
            else LINE_B
        )

        points.append((line_x, y))

        print(f"{names[len(points)-1]}: ({line_x}, {y})")
        redraw()

cv2.namedWindow("Lane Calibration")
cv2.setMouseCallback("Lane Calibration", on_click)

redraw()

print("Select points in this order:")
print("1. Near lane at x=140")
print("2. Near lane at x=500")
print("3. Far lane at x=140")
print("4. Far lane at x=500")
print("R = Reset | S = Save | Q = Quit")

while True:
    key = cv2.waitKey(30) & 0xFF

    if key == ord("r"):
        points.clear()
        print("Reset.")
        redraw()

    elif key == ord("s"):
        if len(points) != 4:
            print("Please select all four points.")
            continue

        data = {
            "near_lane": {
                "point_a": points[0],
                "point_b": points[1]
            },
            "far_lane": {
                "point_a": points[2],
                "point_b": points[3]
            },
            "reference_distance_meters": 16.5,
            "distance_status": "rough_estimate"
        }

        OUTPUT.write_text(json.dumps(data, indent=2))
        print(f"Saved: {OUTPUT.resolve()}")
        break

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
