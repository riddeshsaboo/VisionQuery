import threading
import time
import cv2
from model_manager import ModelManager
from attribute_extractor import AttributeExtractor
from config import PROCESS_HEIGHT, PROCESS_WIDTH, TARGET_FPS, YOLO_IMAGE_SIZE
from event_manager import EventManager
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

        self.ai_fps = TARGET_FPS
        self.ai_interval = 1.0 / self.ai_fps

        self.model = ModelManager.get_model()
        self.track_manager = TrackManager(disappearance_timeout=2.0)
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
        frame_count = 0
        next_inference_time = time.time()

        while self.running:
            frame = self.camera_connection.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Resize only when YOLO inference is actually needed
            h, w = frame.shape[:2]

            if (w != PROCESS_WIDTH or h != PROCESS_HEIGHT):
                frame = cv2.resize(frame,(PROCESS_WIDTH, PROCESS_HEIGHT),interpolation=cv2.INTER_AREA)
            
            # Motion detection check every 5 frames
            frame_count += 1
            if frame_count % 5 == 0:
                motion_frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                mask = self.fgbg.apply(motion_frame)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.motion_kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.motion_detected = any(cv2.contourArea(c) > 100 for c in contours)

            if not self.motion_detected:
                display_frame = frame.copy()
                cv2.putText(display_frame, "NO MOTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                with self.result_lock:
                    self.latest_annotated_frame = display_frame
                time.sleep(0.01)
                continue

            # Rate limit inference
            current_time = time.time()
            if current_time < next_inference_time:
                time.sleep(0.005)
                continue
            next_inference_time = current_time + self.ai_interval

            # YOLO inference and tracking
            with ModelManager.get_lock():
                results = self.model.track(
                    frame,
                    tracker="bytetrack.yaml",
                    persist=True,
                    imgsz=YOLO_IMAGE_SIZE,
                    classes=[0, 2, 3, 5, 7],
                    verbose=False
                )

            # Process completed tracks
            for track in self.track_manager.update(results, frame):
                track["camera_id"] = self.camera_id
                track["attributes"] = self.attribute_extractor.extract(track["best_crop"], track["object_type"])
                self.event_manager.save_event(track)

            # Annotate detections and track IDs
            annotated_frame = results[0].plot(labels=False, conf=True)
            boxes_obj = results[0].boxes
            if boxes_obj.id is not None:
                track_ids = boxes_obj.id.int().cpu().tolist()
                classes = boxes_obj.cls.int().cpu().tolist()
                boxes = boxes_obj.xyxy.cpu().tolist()

                for box, track_id, class_id in zip(boxes, track_ids, classes):
                    x1, y1 = int(box[0]), int(box[1])
                    label = f"ID: {track_id} {self.model.names[class_id]}"
                    cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            with self.result_lock:
                self.latest_annotated_frame = annotated_frame.copy()

    def get_frame(self):
        with self.result_lock:
            return self.latest_annotated_frame.copy() if self.latest_annotated_frame is not None else None

    def stop(self):
        print(f"[VISION {self.camera_id}] Stopping...")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None

        with self.result_lock:
            self.latest_annotated_frame = None
