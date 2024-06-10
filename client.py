import requests
import cv2
from picamera2 import Picamera2
import numpy as np

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"format": "RGB888", "size": (800, 980)})
)
picam2.start()

url = "http://10.42.0.1:5001/"  # Flask server address

while True:
    frame = picam2.capture_array()
    _, img_encoded = cv2.imencode(".jpg", frame)
    requests.post(url + "upload", data=img_encoded.tobytes())
