import sys
import threading
import time
from pathlib import Path
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from vision_processor import VisionProcessor


class CameraConnection:
    def __init__(self, camera_config):
        self.camera_config = camera_config
        self.camera_id = camera_config["id"]
        self.camera_type = camera_config["type"]

        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.status = "offline"

        self.thread = None
        self.vision_processor = None

    def get_source(self):
        if self.camera_type == "webcam":
            return self.camera_config.get("device", 0)

        method = self.camera_config.get("connection_method")
        if method == "rtsp":
            return self.camera_config.get("rtsp_url")
        if method == "credentials":
            return f"rtsp://{self.camera_config.get('username')}:{self.camera_config.get('password')}@{self.camera_config.get('ip')}:554/"
        return None

    def start(self):
        if self.running:
            return
        self.running = True
        self.status = "connecting"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        source = self.get_source()
        if source is None:
            self.status, self.running = "offline", False
            return

        while self.running:
            try:
                print(f"[CAMERA {self.camera_id}] Connecting...")
                self.cap = (
                    cv2.VideoCapture(source)
                    if self.camera_type == "webcam"
                    else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                )

                if not self.cap.isOpened():
                    print(f"[CAMERA {self.camera_id}] Connection failed")
                    self.status = "offline"
                    self._release()
                    time.sleep(3)
                    continue

                print(f"[CAMERA {self.camera_id}] ONLINE")
                self.status = "online"
                self.vision_processor = VisionProcessor(self, self.camera_id)
                self.vision_processor.start()

                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        print(f"[CAMERA {self.camera_id}] Stream lost")
                        self.status = "offline"
                        break
                    with self.lock:
                        self.frame = frame

                self._release()

            except Exception as error:
                print(f"[CAMERA {self.camera_id}] Error: {error}")
                self.status = "offline"
                self._release()
                if self.running:
                    time.sleep(3)

        self._stop_processing()
        self.status = "offline"
        self._release()
        print(f"[CAMERA {self.camera_id}] STOPPED")

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_status(self):
        return self.status

    def _release(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _stop_processing(self):
        if self.vision_processor:
            self.vision_processor.stop()
            self.vision_processor = None

    def stop(self):
        print(f"[CAMERA {self.camera_id}] Stopping...")
        self.running = False
        self._stop_processing()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self._release()
        self.thread = None
        self.status = "offline"
        with self.lock:
            self.frame = None
        print(f"[CAMERA {self.camera_id}] Stopped")