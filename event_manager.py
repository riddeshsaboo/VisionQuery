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
            "object_type": track["object_type"],
            "start_time": self._format_timestamp(track["first_seen"]),
            "end_time": self._format_timestamp(track["last_seen"]),
            "attributes": track.get("attributes"),
            "video_file": None,
        }

        events.append(event)
        with open(self.file_path, "w") as f:
            json.dump(events, f, indent=4)

        self._print_event(event)
        self.next_event_id += 1
        return event

    def _format_timestamp(self, timestamp):
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def _print_event(self, event):
        print(f"\n[EVENT SAVED] ID: {event['event_id']} | Track: {event['track_id']} | Type: {event['object_type']} | {event['start_time']} -> {event['end_time']}\n")