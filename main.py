import os
import threading
import time
import cv2
import psutil
from ultralytics import YOLO

from config import (
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    TARGET_FPS,
    YOLO_IMAGE_SIZE,
    get_device,
)
from track_manager import TrackManager
from event_manager import EventManager
from event_store import EventStore
from attribute_extractor import AttributeExtractor

# =====================================================
# CAMERA / IP STREAM READER
# =====================================================
class IPStreamReader:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.ret = False
        self.frame = None
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self

        print("Connecting to camera...")

        # Local webcam
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)

        # IP / RTSP camera
        else:
            self.cap = cv2.VideoCapture(
                self.source,
                cv2.CAP_FFMPEG
            )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera: {self.source}"
            )

        self.started = True

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )
        self.thread.start()

        print("Camera connected successfully.")

        return self

    def update(self):
        while self.started:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.read_lock:
                self.ret = True
                self.frame = frame

    def read(self):
        with self.read_lock:
            return self.ret, self.frame

    def stop(self):
        self.started = False
        if self.cap.isOpened():
            self.cap.release()

# =====================================================
# CONFIGURATION & INITIALIZATION
# =====================================================
CAMERA_URL = 0        # for laptop's front camera !
# CAMERA_URL = 'rtsp://admin:123456@192.168.1.64:554/'
FRAME_DELAY = 1.0 / TARGET_FPS

device = get_device()
print(f"Using device: {device}")

model = YOLO("yolov8n.pt")
model.to(device)

track_manager = TrackManager(disappearance_timeout=2.0)
event_manager = EventManager("events.json")
attribute_extractor = AttributeExtractor()
event_store = EventStore("events.json")
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

print("Connecting to camera...")
stream = IPStreamReader(CAMERA_URL).start()
time.sleep(2)

process = psutil.Process(os.getpid())
last_stats_time = time.time()

print("Stream Started Successfully...")
print(f"Target YOLO FPS : {TARGET_FPS}")
print(f"Processing Size  : {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
print(f"YOLO Image Size  : {YOLO_IMAGE_SIZE}")

# =====================================================
# MAIN LOOP
# =====================================================
try:
    frame_count = 0
    motion = False
    next_inference_time = time.time()
    inference_count = 0
    inference_start_time = time.time()

    while True:
        loop_start = time.time()

        # Get latest frame
        ret, original_frame = stream.read()
        if not ret or original_frame is None:
            time.sleep(0.05)
            continue

        # Resize for processing if needed
        if original_frame.shape[1] != PROCESS_WIDTH or original_frame.shape[0] != PROCESS_HEIGHT:
            frame = cv2.resize(original_frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            frame = original_frame

        # Motion detection (every 5 frames)
        frame_count += 1
        if frame_count % 5 == 0:
            motion_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            mask = fgbg.apply(motion_frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion = any(cv2.contourArea(c) > 500 for c in contours)

        # Handle No Motion
        if not motion:
            cv2.putText(frame, "NO MOTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("VisionQuery", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            sleep_time = FRAME_DELAY - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            continue

        # Rate-limit YOLO
        current_time = time.time()
        if current_time < next_inference_time:
            cv2.putText(frame, "MOTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("VisionQuery", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(max(0, next_inference_time - time.time()))
            continue

        next_inference_time = time.time() + FRAME_DELAY

        # YOLO + ByteTrack
        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            imgsz=YOLO_IMAGE_SIZE,
            classes=[0, 2, 3, 5, 7],  # person, car, motorcycle, bus, truck
            verbose=False,
        )

        # YOLO FPS tracking
        inference_count += 1
        inference_elapsed = time.time() - inference_start_time

        if inference_elapsed >= 5:
            actual_inference_fps = inference_count / inference_elapsed
            print(f"YOLO FPS: {actual_inference_fps:.2f}")
            inference_count = 0
            inference_start_time = time.time()


        # =====================================================
        # TRACK MANAGER
        # =====================================================

        completed_tracks = track_manager.update(results, frame)

        for track in completed_tracks:

            # ---------------------------------------------
            # Extract attributes from best crop
            # ---------------------------------------------

            attributes = attribute_extractor.extract(
                track["best_crop"],
                track["object_type"]
            )

            track["attributes"] = attributes

            # ---------------------------------------------
            # Save completed event
            # ---------------------------------------------

            event_manager.save_event(track)


        # =====================================================
        # ANNOTATION
        # =====================================================

        annotated_frame = results[0].plot(
            labels=False,
            conf=False
        )

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)

                class_name = model.names[cls_id]

                cv2.putText(
                    annotated_frame,
                    f"ID: {track_id} {class_name}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        # =====================================================
        # PERFORMANCE METRICS
        # =====================================================

        elapsed = time.time() - loop_start

        # Display / processing loop FPS
        display_fps = 1.0 / elapsed if elapsed > 0 else 0

        # YOLO FPS
        yolo_elapsed = time.time() - inference_start_time

        if yolo_elapsed > 0:
            current_yolo_fps = inference_count / yolo_elapsed
        else:
            current_yolo_fps = 0

        cpu = process.cpu_percent(interval=None)
        system_cpu = psutil.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024 * 1024)

        # Draw performance overlays
        cv2.putText(
            annotated_frame,
            f"Display FPS : {display_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"YOLO FPS : {current_yolo_fps:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"CPU : {cpu:.1f}%",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"RAM : {ram:.1f} MB",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        cv2.imshow("VisionQuery", annotated_frame)

        # Console stats
        if time.time() - last_stats_time >= 1:
            print(
                f"Display FPS: {display_fps:.2f} | "
                f"YOLO FPS: {current_yolo_fps:.2f} | "
                f"CPU: {cpu:.1f}% | "
                f"System CPU: {system_cpu:.1f}% | "
                f"RAM: {ram:.1f} MB"
            )
            last_stats_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Maintain target FPS
        sleep_time = FRAME_DELAY - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

finally:
    print("Stopping stream...")
    stream.stop()
    cv2.destroyAllWindows()