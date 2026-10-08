from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

previous_frame = None

while True:
    frame = picam2.capture_array()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame is None:
        previous_frame = gray
        continue

    frame_diff = cv2.absdiff(previous_frame, gray)

    threshold = cv2.threshold(
        frame_diff,
        25,
        255,
        cv2.THRESH_BINARY
    )[1]

    threshold = cv2.dilate(threshold, None, iterations=2)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_found = False

    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            motion_found = True
            x, y, w, h = cv2.boundingRect(contour)

            print(
                f"Motion detected: x={x}, y={y}, "
                f"width={w}, height={h}"
            )

    if motion_found:
        print("Moving object detected")

    previous_frame = gray
