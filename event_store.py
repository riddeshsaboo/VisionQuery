import json
import os
from datetime import datetime


class EventStore:
    def __init__(self, filename="events.json"):
        self.filename = filename
        self.events = self._load()

    def _load(self):
        if not os.path.exists(self.filename):
            return []
        try:
            with open(self.filename, "r") as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def add_event(self, track):
        event = {
            "event_id": len(self.events) + 1,
            "track_id": track["track_id"],
            "object_type": track["object_type"],
            "start_time": self._format_timestamp(track["first_seen"]),
            "end_time": self._format_timestamp(track["last_seen"]),
            "attributes": track.get("attributes"),
            "video_file": None,
        }
        self.events.append(event)
        self._save()
        print(f"[EVENT SAVED] Event ID={event['event_id']} Track ID={event['track_id']} Object={event['object_type']}")
        return event

    def _format_timestamp(self, timestamp):
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def _save(self):
        with open(self.filename, "w") as file:
            json.dump(self.events, file, indent=4)

    def get_events(self):
        return self.events