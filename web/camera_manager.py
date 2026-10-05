import cv2
import threading
import time
from pathlib import Path
import sys
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

    # ========================================================
    # GET SOURCE
    # ========================================================

    def get_source(self):

        if self.camera_type == "webcam":

            return self.camera_config.get(
                "device",
                0
            )

        method = self.camera_config.get(
            "connection_method"
        )

        if method == "rtsp":

            return self.camera_config.get(
                "rtsp_url"
            )

        if method == "credentials":

            ip = self.camera_config.get("ip")
            username = self.camera_config.get("username")
            password = self.camera_config.get("password")

            return (
                f"rtsp://{username}:{password}"
                f"@{ip}:554/"
            )

        return None

    # ========================================================
    # START
    # ========================================================

    def start(self):

        if self.running:
            return

        self.running = True
        self.status = "connecting"

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    def _run(self):

        source = self.get_source()

        if source is None:

            self.status = "offline"
            self.running = False

            return

        while self.running:

            try:

                print(
                    f"[CAMERA {self.camera_id}] Connecting..."
                )

                if self.camera_type == "webcam":

                    self.cap = cv2.VideoCapture(
                        source
                    )

                else:

                    self.cap = cv2.VideoCapture(
                        source,
                        cv2.CAP_FFMPEG
                    )

                if not self.cap.isOpened():

                    print(
                        f"[CAMERA {self.camera_id}] "
                        f"Connection failed"
                    )

                    self.status = "offline"

                    self._release()

                    time.sleep(3)

                    continue

                print(f"[CAMERA {self.camera_id}] ONLINE")
                self.status = "online"

                # Start VisionQuery AI processing
                self.vision_processor = VisionProcessor(
                    self,
                    self.camera_id
                )

                self.vision_processor.start()

                # ---------------------------------------------
                # READ FRAMES
                # ---------------------------------------------

                while self.running:

                    ret, frame = self.cap.read()

                    if not self.running:
                        break

                    if not ret:

                        print(
                            f"[CAMERA {self.camera_id}] "
                            f"Stream lost"
                        )

                        self.status = "offline"

                        break

                    with self.lock:

                        self.frame = frame

                self._release()

            except Exception as e:

                print(
                    f"[CAMERA {self.camera_id}] "
                    f"Error: {e}"
                )

                self.status = "offline"

                self._release()

                if self.running:

                    time.sleep(3)

        self.status = "offline"

        self._release()

        print(
            f"[CAMERA {self.camera_id}] STOPPED"
        )

    # ========================================================
    # GET FRAME
    # ========================================================

    def get_frame(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        return self.status

    # ========================================================
    # RELEASE
    # ========================================================

    def _release(self):

        if self.cap is not None:

            try:
                self.cap.release()
            except Exception:
                pass

            self.cap = None

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):
        print(f"[CAMERA {self.camera_id}] Stopping...")

        if self.vision_processor is not None:
            self.vision_processor.stop()
            self.vision_processor = None

        self.running = False

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self._release()

        self.thread = None
        self.status = "offline"

        with self.lock:
            self.frame = None

        print(f"[CAMERA {self.camera_id}] Stopped")