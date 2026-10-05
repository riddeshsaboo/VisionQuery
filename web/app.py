from flask import Flask, render_template, request, jsonify, Response
import json
import os
import psutil
import platform
from camera_manager import CameraConnection
import cv2
import time

try:
    import torch
except ImportError:
    torch = None


app = Flask(__name__)

camera_connections = {}
CAMERAS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "cameras.json"
)


# ============================================================
# CAMERA STORAGE
# ============================================================

def load_cameras():

    if not os.path.exists(CAMERAS_FILE):
        return []

    try:
        with open(CAMERAS_FILE, "r") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return []

def start_all_cameras():

    cameras = load_cameras()

    for camera in cameras:

        camera_id = camera["id"]

        connection = CameraConnection(
            camera
        )

        connection.start()

        camera_connections[camera_id] = connection

        print(
            f"[CAMERA {camera_id}] "
            f"Starting: {camera['name']}"
        )


def save_cameras(cameras):

    with open(CAMERAS_FILE, "w") as file:
        json.dump(cameras, file, indent=4)


# ============================================================
# SYSTEM CAPACITY
# ============================================================

def get_system_capacity():

    logical_cores = psutil.cpu_count(logical=True) or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    system = platform.system()
    machine = platform.machine()

    cuda_available = False
    mps_available = False

    added_cameras = 0 

    if torch is not None:

        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        try:
            mps_available = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except Exception:
            mps_available = False

    # --------------------------------------------------------
    # Base capacity
    # --------------------------------------------------------
    #
    # This is intentionally conservative.
    #
    # We are NOT assuming:
    #
    # 1 CPU core = 1 camera
    #
    # YOLO, tracking, decoding, motion detection, etc.
    # all consume different resources.
    #

    if cuda_available or mps_available:

        if physical_cores >= 10 and ram_gb >= 16:
            max_cameras = 8

        elif physical_cores >= 8 and ram_gb >= 16:
            max_cameras = 6

        elif physical_cores >= 6 and ram_gb >= 16:
            max_cameras = 4

        elif physical_cores >= 4 and ram_gb >= 8:
            max_cameras = 3

        else:
            max_cameras = 2

    else:

        if physical_cores >= 12 and ram_gb >= 16:
            max_cameras = 5

        elif physical_cores >= 8 and ram_gb >= 16:
            max_cameras = 4

        elif physical_cores >= 6 and ram_gb >= 8:
            max_cameras = 3

        elif physical_cores >= 4 and ram_gb >= 8:
            max_cameras = 2

        else:
            max_cameras = 1

    # Never allow more cameras than logical cores.
    max_cameras = min(
        max_cameras,
        logical_cores
    )

    max_cameras = max(
        1,
        max_cameras
    )

    if cuda_available:
        accelerator = "NVIDIA CUDA"

    elif mps_available:
        accelerator = "Apple MPS"

    else:
        accelerator = "CPU"


    try:
        with open(CAMERAS_FILE, "r") as file:
            added_cameras = len(json.load(file))

    except (json.JSONDecodeError, OSError):
        pass

    return {
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "ram_gb": round(ram_gb, 1),
        "os": system,
        "architecture": machine,
        "accelerator": accelerator,
        "max_cameras": max_cameras, 
        "added_cameras": added_cameras
    }

def generate_camera_frames(camera_id):
    while True:
        connection = camera_connections.get(camera_id)

        if connection is None:
            break

        # Get YOLO processed frame
        if connection.vision_processor is not None:
            frame = connection.vision_processor.get_frame()
        else:
            frame = connection.get_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.1)

# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# SYSTEM INFO
# ============================================================

@app.route("/api/system")
def system_info(): 
    
    return jsonify(
        get_system_capacity()
    ) 


# ============================================================
# GET CAMERAS
# ============================================================

@app.route(
    "/api/cameras",
    methods=["GET"]
)
@app.route("/api/cameras",methods=["GET"])
def get_cameras():

    cameras = load_cameras()


    for camera in cameras:

        camera_id = camera["id"]


        if camera_id in camera_connections:

            camera["status"] = (
                camera_connections[
                    camera_id
                ].get_status()
            )

        else:

            camera["status"] = "offline"


    return jsonify(cameras)

# ============================================================
# ADD CAMERA
# ============================================================

@app.route(
    "/api/cameras",
    methods=["POST"]
)
def add_camera():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Invalid data"
        }), 400

    name = data.get(
        "name",
        ""
    ).strip()

    camera_type = data.get(
        "type",
        ""
    ).strip()

    if not name:

        return jsonify({
            "error": "Camera name is required"
        }), 400

    if camera_type not in [
        "ip",
        "webcam"
    ]:

        return jsonify({
            "error": "Invalid camera type"
        }), 400

    cameras = load_cameras()

    # --------------------------------------------------------
    # CAMERA LIMIT
    # --------------------------------------------------------

    capacity = get_system_capacity()

    if len(cameras) >= capacity["max_cameras"]:

        return jsonify({
            "error": (
                f"Maximum camera limit reached. "
                f"This machine is currently configured "
                f"for up to {capacity['max_cameras']} cameras."
            )
        }), 409

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if cameras:

        camera_id = max(
            camera["id"]
            for camera in cameras
        ) + 1

    else:

        camera_id = 1

    # --------------------------------------------------------
    # IP CAMERA
    # --------------------------------------------------------

    if camera_type == "ip":

        connection_method = data.get(
            "connection_method"
        )

        if connection_method not in [
            "rtsp",
            "credentials"
        ]:

            return jsonify({
                "error": "Invalid connection method"
            }), 400

        camera = {

            "id": camera_id,

            "name": name,

            "type": "ip",

            "connection_method":
                connection_method,

            "rtsp_url": "",

            "ip": "",

            "username": "",

            "password": "",

            "nvr_channel": None,

            "status": "offline"
        }

        # ----------------------------------------------------
        # DIRECT RTSP
        # ----------------------------------------------------

        if connection_method == "rtsp":

            rtsp_url = data.get(
                "rtsp_url",
                ""
            ).strip()

            if not rtsp_url:

                return jsonify({
                    "error":
                        "RTSP URL is required"
                }), 400

            camera["rtsp_url"] = rtsp_url

        # ----------------------------------------------------
        # IP + CREDENTIALS
        # ----------------------------------------------------

        else:

            ip = data.get(
                "ip",
                ""
            ).strip()

            username = data.get(
                "username",
                ""
            ).strip()

            password = data.get(
                "password",
                ""
            )

            if not ip:

                return jsonify({
                    "error":
                        "IP address is required"
                }), 400

            if not username:

                return jsonify({
                    "error":
                        "Username is required"
                }), 400

            if not password:

                return jsonify({
                    "error":
                        "Password is required"
                }), 400

            camera["ip"] = ip
            camera["username"] = username
            camera["password"] = password

        # ----------------------------------------------------
        # NVR CHANNEL
        # ----------------------------------------------------

        nvr_channel = data.get(
            "nvr_channel"
        )

        try:

            nvr_channel = int(
                nvr_channel
            )

            if nvr_channel < 1:

                raise ValueError

        except (
            TypeError,
            ValueError
        ):

            return jsonify({
                "error":
                    "Valid NVR channel number is required"
            }), 400

        camera["nvr_channel"] = nvr_channel

    # --------------------------------------------------------
    # WEBCAM
    # --------------------------------------------------------

    else:

        device = data.get(
            "device",
            0
        )

        try:

            device = int(device)

            if device < 0:
                raise ValueError

        except (
            TypeError,
            ValueError
        ):

            return jsonify({
                "error":
                    "Invalid webcam device"
            }), 400

        camera = {

            "id": camera_id,

            "name": name,

            "type": "webcam",

            "device": device,

            "status": "offline"
        }

    cameras.append(
        camera
    )

    save_cameras(
        cameras
    )

    connection = CameraConnection(camera)
    connection.start()
    camera_connections[camera_id] = connection

    return jsonify(
        camera
    ), 201


# ============================================================
# DELETE CAMERA
# ============================================================

@app.route(
    "/api/cameras/<int:camera_id>",
    methods=["DELETE"]
)
def delete_camera(camera_id):

    cameras = load_cameras()

    camera = next(
        (
            camera
            for camera in cameras
            if camera["id"] == camera_id
        ),
        None
    )

    if camera is None:

        return jsonify({
            "error": "Camera not found"
        }), 404


    # STOP CAMERA

    if camera_id in camera_connections:

        connection = camera_connections[
            camera_id
        ]

        connection.stop()

        del camera_connections[
            camera_id
        ]


    # REMOVE FROM JSON

    cameras = [
        camera
        for camera in cameras
        if camera["id"] != camera_id
    ]


    save_cameras(cameras)


    return jsonify({
        "success": True,
        "message": "Camera removed"
    })



@app.route(
    "/video_feed/<int:camera_id>"
)
def video_feed(camera_id):

    if camera_id not in camera_connections:

        return (
            "Camera not available",
            404
        )

    return Response(
        generate_camera_frames(
            camera_id
        ),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    start_all_cameras()

    app.run(
        debug=True, use_reloader=False
    )