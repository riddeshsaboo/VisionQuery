import os
import threading
import time
import cv2
import psutil
from ultralytics import YOLO

from attribute_extractor import AttributeExtractor
from config import PROCESS_HEIGHT, PROCESS_WIDTH, TARGET_FPS, YOLO_IMAGE_SIZE, get_device
from event_manager import EventManager
from event_store import EventStore
from track_manager import TrackManager


# Camera / RTSP stream reader thread
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
        self.cap = (
            cv2.VideoCapture(self.source)
            if isinstance(self.source, int)
            else cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        )
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera: {self.source}")
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
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
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.started = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


# Config & Pipeline Setup
# CAMERA_URL = 0        # for laptop's front camera !
CAMERA_URL = 'rtsp://admin:123456@192.168.0.50:554/'
DISPLAY_FPS = 30
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS
AI_FPS = TARGET_FPS
AI_INTERVAL = 1.0 / AI_FPS

device = get_device()
print(f"Using device: {device}")
model = YOLO("yolov8n.pt").to(device)

track_manager = TrackManager(disappearance_timeout=2.0)
event_manager = EventManager("events.json")
attribute_extractor = AttributeExtractor()
event_store = EventStore("events.json")

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
motion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Thread synchronization & telemetry
latest_annotated_frame = None
motion_detected = False
result_lock = threading.Lock()
stop_event = threading.Event()

process = psutil.Process(os.getpid())
last_stats_time = time.time()
display_fps = 0.0
current_yolo_fps = 0.0
cpu, system_cpu, ram = 0.0, 0.0, 0.0


# AI processing loop
def ai_processing_loop(stream):
    global latest_annotated_frame, motion_detected, current_yolo_fps, cpu, system_cpu, ram

    frame_count = 0
    inference_count = 0
    inference_start = time.time()
    last_motion_check = False
    next_inference_time = time.time()

    print("AI processing thread started.")

    while not stop_event.is_set():
        ret, original_frame = stream.read()
        if not ret or original_frame is None:
            time.sleep(0.01)
            continue

        # Resize for inference
        h, w = original_frame.shape[:2]
        frame = cv2.resize(original_frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA) if (w != PROCESS_WIDTH or h != PROCESS_HEIGHT) else original_frame

        # Motion detection check every 5 frames
        frame_count += 1
        if frame_count % 5 == 0:
            motion_frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            mask = fgbg.apply(motion_frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, motion_kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            last_motion_check = any(cv2.contourArea(c) > 100 for c in contours)
        motion_detected = last_motion_check

        if frame_count % 30 == 0:
            print(f"MOTION STATUS: {motion_detected}")

        if not motion_detected:
            with result_lock:
                display_frame = frame.copy()
                cv2.putText(display_frame, "NO MOTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                latest_annotated_frame = display_frame
            time.sleep(0.01)
            continue

        # Rate limit inference
        curr_time = time.time()
        if curr_time < next_inference_time:
            time.sleep(0.005)
            continue
        next_inference_time = curr_time + AI_INTERVAL

        # YOLO inference & tracking
        results = model.track(
            frame, tracker="bytetrack.yaml", persist=True,
            imgsz=YOLO_IMAGE_SIZE, classes=[0, 2, 3, 5, 7], verbose=False
        )

        # Update YOLO FPS
        inference_count += 1
        elapsed = time.time() - inference_start
        if elapsed >= 5.0:
            current_yolo_fps = inference_count / elapsed
            print(f"YOLO FPS: {current_yolo_fps:.2f}")
            inference_count, inference_start = 0, time.time()

        # Update tracks & extract attributes
        for track in track_manager.update(results, frame):
            track["attributes"] = attribute_extractor.extract(track["best_crop"], track["object_type"])
            event_manager.save_event(track)

        # Annotate bounding boxes and IDs
        annotated_frame = results[0].plot(labels=False, conf=True)
        boxes_obj = results[0].boxes
        if boxes_obj.id is not None:
            t_ids = boxes_obj.id.int().cpu().tolist()
            classes = boxes_obj.cls.int().cpu().tolist()
            boxes = boxes_obj.xyxy.cpu().tolist()

            for box, tid, cid in zip(boxes, t_ids, classes):
                x1, y1, _, _ = map(int, box)
                cv2.putText(
                    annotated_frame, f"ID: {tid} {model.names[cid]}",
                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                )

        with result_lock:
            latest_annotated_frame = annotated_frame.copy()

        # # Update telemetry
        # cpu = process.cpu_percent(interval=None)
        # system_cpu = psutil.cpu_percent(interval=None)
        # ram = process.memory_info().rss / (1024 * 1024)


# Display thread / Main loop
print("Connecting to camera...")
stream = IPStreamReader(CAMERA_URL).start()
time.sleep(2)

print(f"Stream Started | AI Target: {AI_FPS} FPS | Display Target: {DISPLAY_FPS} FPS")
print(f"Processing Size: {PROCESS_WIDTH}x{PROCESS_HEIGHT} | YOLO Size: {YOLO_IMAGE_SIZE}")

threading.Thread(target=ai_processing_loop, args=(stream,), daemon=True).start()

display_frame_count = 0
display_fps_start = time.time()

try:
    last_metrics_time = time.time()
    process.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)
    
    while True:

        if time.time() - last_metrics_time >= 1:
            cpu = process.cpu_percent(interval=None)
            system_cpu = psutil.cpu_percent(interval=None)
            ram = (process.memory_info().rss / (1024 * 1024))

            last_metrics_time = time.time()

        loop_start = time.time()
        ret, original_frame = stream.read()
        if not ret or original_frame is None:
            time.sleep(0.01)
            continue

        h, w = original_frame.shape[:2]
        frame = cv2.resize(original_frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA) if (w != PROCESS_WIDTH or h != PROCESS_HEIGHT) else original_frame

        with result_lock:
            display_frame = latest_annotated_frame.copy() if latest_annotated_frame is not None else frame.copy()

        # Update display FPS
        display_frame_count += 1
        disp_elapsed = time.time() - display_fps_start
        if disp_elapsed >= 1.0:
            display_fps = display_frame_count / disp_elapsed
            display_frame_count, display_fps_start = 0, time.time()

        # Telemetry overlay
        metrics = [
            (f"Display FPS : {display_fps:.1f}", (0, 255, 0)),
            (f"YOLO FPS : {current_yolo_fps:.2f}", (255, 255, 0)),
            (f"CPU : {cpu:.1f}%", (255, 255, 0)),
            (f"RAM : {ram:.1f} MB", (0, 255, 255)),
        ]
        for idx, (label, color) in enumerate(metrics):
            cv2.putText(display_frame, label, (20, 35 + (idx * 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("VisionQuery", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Log system stats every second
        if time.time() - last_stats_time >= 1:
            print(f"Display FPS: {display_fps:.2f} | YOLO FPS: {current_yolo_fps:.2f} | CPU: {cpu:.1f}% | System CPU: {system_cpu:.1f}% | RAM: {ram:.1f} MB")
            last_stats_time = time.time()

        # Sync display FPS
        sleep_time = DISPLAY_INTERVAL - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

finally:
    print("Stopping VisionQuery...")
    stop_event.set()
    stream.stop()
    cv2.destroyAllWindows()
    print("VisionQuery stopped.")