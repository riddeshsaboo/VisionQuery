import json
import os
import platform
import time
import cv2
import psutil
from flask import Flask, Response, jsonify, render_template, request

from camera_manager import CameraConnection

try:
    import torch
except ImportError:
    torch = None

app = Flask(__name__)
camera_connections = {}
CAMERAS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cameras.json")


# Storage & startup
def load_cameras():
    if not os.path.exists(CAMERAS_FILE):
        return []
    try:
        with open(CAMERAS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_cameras(cameras):
    with open(CAMERAS_FILE, "w") as f:
        json.dump(cameras, f, indent=4)


def start_all_cameras():
    for cam in load_cameras():
        cid = cam["id"]
        conn = CameraConnection(cam)
        conn.start()
        camera_connections[cid] = conn
        print(f"[CAMERA {cid}] Starting: {cam['name']}")


# Hardware capacity sizing
def get_system_capacity():
    logical_cores = psutil.cpu_count(logical=True) or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    cuda_avail = bool(torch and torch.cuda.is_available())
    mps_avail = bool(torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    if cuda_avail or mps_avail:
        if physical_cores >= 10 and ram_gb >= 16: max_cams = 8
        elif physical_cores >= 8 and ram_gb >= 16: max_cams = 6
        elif physical_cores >= 6 and ram_gb >= 16: max_cams = 4
        elif physical_cores >= 4 and ram_gb >= 8: max_cams = 3
        else: max_cams = 2
    else:
        if physical_cores >= 12 and ram_gb >= 16: max_cams = 5
        elif physical_cores >= 8 and ram_gb >= 16: max_cams = 4
        elif physical_cores >= 6 and ram_gb >= 8: max_cams = 3
        elif physical_cores >= 4 and ram_gb >= 8: max_cams = 2
        else: max_cams = 1

    max_cams = max(1, min(max_cams, logical_cores))
    accelerator = "NVIDIA CUDA" if cuda_avail else ("Apple MPS" if mps_avail else "CPU")

    return {
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "ram_gb": round(ram_gb, 1),
        "os": platform.system(),
        "architecture": platform.machine(),
        "accelerator": accelerator,
        "max_cameras": max_cams,
        "added_cameras": len(load_cameras()),
    }


# MJPEG video feed generator
def generate_camera_frames(camera_id):
    while True:
        conn = camera_connections.get(camera_id)
        if conn is None:
            break

        frame = conn.vision_processor.get_frame() if conn.vision_processor else conn.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.1)


# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/system")
def system_info():
    return jsonify(get_system_capacity())


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    cameras = load_cameras()
    for cam in cameras:
        conn = camera_connections.get(cam["id"])
        cam["status"] = conn.get_status() if conn else "offline"
    return jsonify(cameras)


@app.route("/api/cameras", methods=["POST"])
def add_camera():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    cam_type = data.get("type", "").strip()

    if not name:
        return jsonify({"error": "Camera name is required"}), 400
    if cam_type not in {"ip", "webcam"}:
        return jsonify({"error": "Invalid camera type"}), 400

    cameras = load_cameras()
    capacity = get_system_capacity()
    if len(cameras) >= capacity["max_cameras"]:
        return jsonify({"error": f"Max camera limit reached ({capacity['max_cameras']})."}), 409

    cid = (max(c["id"] for c in cameras) + 1) if cameras else 1
    camera = _build_ip_camera(data, cid, name) if cam_type == "ip" else _build_webcam_camera(data, cid, name)

    if isinstance(camera, tuple):
        return camera

    cameras.append(camera)
    save_cameras(cameras)

    conn = CameraConnection(camera)
    conn.start()
    camera_connections[cid] = conn
    return jsonify(camera), 201


def _build_ip_camera(data, cid, name):
    method = data.get("connection_method")
    if method not in {"rtsp", "credentials"}:
        return jsonify({"error": "Invalid connection method"}), 400

    cam = {"id": cid, "name": name, "type": "ip", "connection_method": method, "status": "offline"}

    if method == "rtsp":
        url = data.get("rtsp_url", "").strip()
        if not url:
            return jsonify({"error": "RTSP URL is required"}), 400
        cam.update({"rtsp_url": url, "ip": "", "username": "", "password": ""})
    else:
        ip, user, pwd = data.get("ip", "").strip(), data.get("username", "").strip(), data.get("password", "")
        if not ip or not user or not pwd:
            return jsonify({"error": "IP, username, and password are required"}), 400
        cam.update({"rtsp_url": "", "ip": ip, "username": user, "password": pwd})

    try:
        nvr_ch = int(data.get("nvr_channel"))
        if nvr_ch < 1: raise ValueError
        cam["nvr_channel"] = nvr_ch
    except (TypeError, ValueError):
        return jsonify({"error": "Valid NVR channel number is required"}), 400

    return cam


def _build_webcam_camera(data, cid, name):
    try:
        dev = int(data.get("device", 0))
        if dev < 0: raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid webcam device"}), 400

    return {"id": cid, "name": name, "type": "webcam", "device": dev, "status": "offline"}


@app.route("/api/cameras/<int:camera_id>", methods=["DELETE"])
def delete_camera(camera_id):
    cameras = load_cameras()
    if not any(c["id"] == camera_id for c in cameras):
        return jsonify({"error": "Camera not found"}), 404

    conn = camera_connections.pop(camera_id, None)
    if conn:
        conn.stop()

    save_cameras([c for c in cameras if c["id"] != camera_id])
    return jsonify({"success": True, "message": "Camera removed"})


@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    if camera_id not in camera_connections:
        return "Camera not available", 404
    return Response(generate_camera_frames(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    start_all_cameras()
    app.run(debug=True, use_reloader=False)