from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)

background = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=40,
    detectShadows=False
)

picam2.start()
time.sleep(2)

print("Learning background. Keep the camera still.")

try:
    # Learn the stationary scene first
    for i in range(90):
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        background.apply(gray)

    print("Ready! Move your hand in front of the camera.")

    saved_count = 0
    last_save = 0

    while saved_count < 5:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        mask = background.apply(gray, learningRate=0.001)

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected = 0

        for contour in contours:
            if cv2.contourArea(contour) > 1500:
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2
                )
                detected += 1


        if detected > 0 and time.monotonic() - last_save >= 0.5:
            saved_count += 1
            filename = f"motion_frame_{saved_count}.jpg"
            cv2.imwrite(
                filename,
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            print(f"Saved {filename}")
            last_save = time.monotonic()

finally:
    picam2.stop()
