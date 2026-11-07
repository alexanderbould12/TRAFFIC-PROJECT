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
saved = False

try:
    while not saved:
        frame = picam2.capture_array()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if previous_frame is None:
            previous_frame = gray
            continue

        difference = cv2.absdiff(previous_frame, gray)
        threshold = cv2.threshold(
            difference, 25, 255, cv2.THRESH_BINARY
        )[1]
        threshold = cv2.dilate(threshold, None, iterations=2)

        contours, _ = cv2.findContours(
            threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected = 0
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 3
                )
                detected += 1

        if detected > 0:
            cv2.imwrite(
                "motion_annotated.jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            print(f"Saved motion_annotated.jpg ({detected} regions)")
            saved = True

        previous_frame = gray

except KeyboardInterrupt:
    print("Stopped without detecting motion.")
finally:
    picam2.stop()
