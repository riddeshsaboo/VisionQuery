import json
import os
from datetime import datetime


class EventStore:

    def __init__(self, filename="events.json"):

        self.filename = filename

        # Load existing events
        if os.path.exists(self.filename):

            try:
                with open(self.filename, "r") as f:
                    self.events = json.load(f)

            except (json.JSONDecodeError, FileNotFoundError):
                self.events = []

        else:
            self.events = []

    # -------------------------------------------------
    # Add completed track
    # -------------------------------------------------

    def add_event(self, track):

        event = {

            "event_id": len(self.events) + 1,

            "track_id": track["track_id"],

            "object_type": track["object_type"],

            "start_time": datetime.fromtimestamp(
                track["first_seen"]
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "end_time": datetime.fromtimestamp(
                track["last_seen"]
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "attributes": track.get(
                "attributes",
                None
            ),

            "video_file": None
        }

        self.events.append(event)

        self._save()

        print(
            f"[EVENT SAVED] "
            f"Event ID={event['event_id']} "
            f"Track ID={event['track_id']} "
            f"Object={event['object_type']}"
        )

        return event

    # -------------------------------------------------
    # Save JSON
    # -------------------------------------------------

    def _save(self):

        with open(self.filename, "w") as f:

            json.dump(
                self.events,
                f,
                indent=4
            )

    # -------------------------------------------------
    # Get all events
    # -------------------------------------------------

    def get_events(self):

        return self.events