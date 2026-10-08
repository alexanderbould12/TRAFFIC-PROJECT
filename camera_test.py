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

frame = picam2.capture_array()

print("Frame shape:", frame.shape)

cv2.imwrite("python_test.jpg", frame)

picam2.stop()

print("Saved python_test.jpg")

