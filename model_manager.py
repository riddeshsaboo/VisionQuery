import threading
from ultralytics import YOLO
from config import get_device


class ModelManager:
    _model = None
    _device = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    cls._device = get_device()
                    print(f"[MODEL] Loading YOLO model on " , f"{cls._device}...")
                    cls._model = YOLO("yolov8n.pt").to(cls._device)
                    print("[MODEL] YOLO model loaded.")

        return cls._model

    @classmethod
    def get_lock(cls):
        return cls._lock