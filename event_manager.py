import json
import os
from datetime import datetime


class EventManager:
    def __init__(self, file_path="events.json"):
        self.file_path = file_path
        self.next_event_id = 1
        self._ensure_file_exists()

        events = self.load_events()
        if events:
            self.next_event_id = max(e["event_id"] for e in events) + 1

    def _ensure_file_exists(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f, indent=4)

    def load_events(self):
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_event(self, track):
        events = self.load_events()
        event = {
            "event_id": self.next_event_id,
            "track_id": track["track_id"],
            "camera_id": track.get("camera_id"),
            "camera_name": track.get("camera_name"),
            "nvr_ip": track.get("nvr_ip"),
            "nvr_channel": track.get("nvr_channel"),
            "object_type": track["object_type"],
            "start_time": self._format_time(track, "first_media_time"),
            "end_time": self._format_time(track, "last_media_time"),
            "attributes": track.get("attributes", {}),
            "video_file": track.get("video_file"),
        }

        events.append(event)
        with open(self.file_path, "w") as f:
            json.dump(events, f, indent=4)

        self._print_event(event)
        self.next_event_id += 1
        return event

    def _format_time(self, track, key):
        val = track.get(key)
        if track.get("source_type") == "video":
            s = max(0, int(float(val or 0)))
            return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S") if val is not None else None

    def _print_event(self, event):
        print(f"\n[EVENT SAVED] ID: {event['event_id']} | Track: {event['track_id']} | Cam: {event['camera_id']} | Type: {event['object_type']} | {event['start_time']} -> {event['end_time']} | File: {event['video_file']}\n")