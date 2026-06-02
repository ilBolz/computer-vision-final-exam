"""OpenCV VideoCapture wrapper."""

import sys
import cv2


class WebcamStreamer:
    def __init__(self, source=0, width=1280, height=720, fps=30):
        """Open video source."""
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        # Su Windows l'auto-backend per webcam USB (DirectShow) è molto lento
        # ad aprirsi. Forziamo CAP_DSHOW quando siamo su Windows e la sorgente
        # è un indice intero (webcam).
        if sys.platform == "win32" and isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {self.source}")

        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        return self

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
