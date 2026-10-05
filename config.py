DEVICE = "auto"

# =====================================================
# CAMERA / PROCESSING CONFIGURATION
# =====================================================

PROCESS_WIDTH = 704
PROCESS_HEIGHT = 576

TARGET_FPS = 3

# YOLO inference resolution
YOLO_IMAGE_SIZE = 640

# =====================================================
# DEVICE SELECTION
# =====================================================

import torch


def get_device():

    if DEVICE != "auto":
        return DEVICE

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"