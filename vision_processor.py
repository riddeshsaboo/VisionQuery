import threading
import time
import cv2

from attribute_extractor import AttributeExtractor
from config import PROCESS_HEIGHT, PROCESS_WIDTH, TARGET_FPS, YOLO_IMAGE_SIZE
from event_manager import EventManager
from model_manager import ModelManager
from track_manager import TrackManager


class VisionProcessor:
    def __init__(self, camera_connection, camera_id):
        self.camera_connection = camera_connection
        self.camera_id = camera_id
        self.running = False
        self.thread = None
        self.latest_annotated_frame = None
        self.result_lock = threading.Lock()
        self.motion_detected = False
        self.frame_count = 0
        self.ai_interval = 1.0 / TARGET_FPS
        self.next_inference_time = time.time()

        self.model = ModelManager.get_model()
        self.track_manager = TrackManager(disappearance_timeout=1.0, min_confirm_frames=2)
        self.event_manager = EventManager("events.json")
        self.attribute_extractor = AttributeExtractor()

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.motion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        print(f"[VISION {self.camera_id}] Processing started.")

    def _processing_loop(self):
        while self.running:
            frame = self.camera_connection.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            self._check_motion(frame)

            if not self.motion_detected:
                self._update_display(frame, "NO MOTION")
                time.sleep(0.01)
                continue

            current_time = time.time()
            if current_time < self.next_inference_time:
                time.sleep(0.005)
                continue
            self.next_inference_time = current_time + self.ai_interval

            inference_frame = self._prepare_inference_frame(frame)
            results = self._run_inference(inference_frame)
            self._process_tracks(results, inference_frame)
            self._update_detection_display(results)

    def _check_motion(self, frame):
        if self.frame_count % 5 != 0:
            return

        motion_frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        mask = self.fgbg.apply(motion_frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.motion_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.motion_detected = any(cv2.contourArea(c) > 50 for c in contours)

    def _prepare_inference_frame(self, frame):
        h, w = frame.shape[:2]
        if w == PROCESS_WIDTH and h == PROCESS_HEIGHT:
            return frame
        return cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)

    def _run_inference(self, frame):
        with ModelManager.get_lock():
            return self.model.track(frame,tracker="bytetrack.yaml",persist=True,imgsz=YOLO_IMAGE_SIZE,classes=[0, 2, 3, 5, 7],verbose=False)

    def _process_tracks(self, results, frame):
        for track in self.track_manager.update(results, frame):
            track["camera_id"] = self.camera_id
            track["attributes"] = self.attribute_extractor.extract(track["best_crop"], track["object_type"])
            self.event_manager.save_event(track)

    def _update_display(self, frame, message):
        display_frame = self._prepare_inference_frame(frame)
        cv2.putText(display_frame,message,(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2)
        with self.result_lock:
            self.latest_annotated_frame = display_frame

    def _update_detection_display(self, results):
        annotated_frame = results[0].plot(labels=False, conf=True)
        boxes = results[0].boxes

        if boxes.id is not None:
            t_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            coords = boxes.xyxy.cpu().tolist()

            for box, tid, cid in zip(coords, t_ids, classes):
                x1, y1 = int(box[0]), int(box[1])
                cv2.putText(annotated_frame,f"ID: {tid} {self.model.names[cid]}",(x1, max(y1 - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0),2)

        with self.result_lock:
            self.latest_annotated_frame = annotated_frame

    def get_frame(self):
        with self.result_lock:
            if self.latest_annotated_frame is not None : 
                return self.latest_annotated_frame.copy()
            else : 
                return None 

    def stop(self):
        print(f"[VISION {self.camera_id}] Stopping...")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        with self.result_lock:
            self.latest_annotated_frame = None