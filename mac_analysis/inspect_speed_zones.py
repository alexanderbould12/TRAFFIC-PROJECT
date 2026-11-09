import cv2
from pathlib import Path

image = cv2.imread("speed_calibration.jpg")

if image is None:
    raise FileNotFoundError("speed_calibration.jpg")

# Our current experimental speed-measurement lines
for x, name in [(140, "A"), (500, "B")]:
    cv2.line(
        image, (x, 0), (x, image.shape[0]),
        (0, 255, 255), 2
    )

    cv2.putText(
        image, f"Line {name}",
        (x - 35, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 2
    )

output = Path("speed_zones_review.jpg")
cv2.imwrite(str(output), image)

print(f"Saved: {output.resolve()}")
