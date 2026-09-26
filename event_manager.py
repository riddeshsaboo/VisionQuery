import json
import os
from datetime import datetime


class EventManager:

    def __init__(self, file_path="events.json"):
        self.file_path = file_path
        self.next_event_id = 1

        # Create file if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f, indent=4)

        # Find next event ID
        events = self.load_events()

        if events:
            self.next_event_id = max(
                event["event_id"] for event in events
            ) + 1

    # -------------------------------------------------
    # Load events
    # -------------------------------------------------

    def load_events(self):

        try:
            with open(self.file_path, "r") as f:
                return json.load(f)

        except (FileNotFoundError, json.JSONDecodeError):
            return []

    # -------------------------------------------------
    # Save event
    # -------------------------------------------------

    def save_event(self, track):

        events = self.load_events()

        event = {
            "event_id": self.next_event_id,
            "track_id": track["track_id"],
            "object_type": track["object_type"],

            "start_time": datetime.fromtimestamp(
                track["first_seen"]
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "end_time": datetime.fromtimestamp(
                track["last_seen"]
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "attributes": track.get("attributes"),

            "video_file": None
        }

        events.append(event)

        with open(self.file_path, "w") as f:
            json.dump(events, f, indent=4)

        print("\n==============================")
        print("EVENT SAVED")
        print(f"Event ID     : {event['event_id']}")
        print(f"Track ID     : {event['track_id']}")
        print(f"Object       : {event['object_type']}")
        print(f"Start        : {event['start_time']}")
        print(f"End          : {event['end_time']}")
        print("==============================\n")

        self.next_event_id += 1

        return event