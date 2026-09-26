import time
import cv2


class TrackManager:
    def __init__(self, disappearance_timeout=2.0):
        self.disappearance_timeout = disappearance_timeout
        self.active_tracks = {}
        self.next_event_id = 1
        self.min_confirm_frames = 3

    def update(self, results, frame):
        current_time = time.time()
        current_track_ids = set()

        if not results:
            return []

        result = results[0]

        # No tracked objects
        if result.boxes.id is None:
            return self._remove_missing_tracks(current_track_ids, current_time)

        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xyxy.int().cpu().tolist()

        # Process detected objects
        for track_id, class_id, box in zip(track_ids, classes, boxes):
            track_id = int(track_id)
            class_id = int(class_id)
            current_track_ids.add(track_id)
            object_type = result.names[class_id]

            # Clamp bounding box to frame dimensions
            height, width = frame.shape[:2]
            x1 = max(0, min(box[0], width - 1))
            y1 = max(0, min(box[1], height - 1))
            x2 = max(0, min(box[2], width - 1))
            y2 = max(0, min(box[3], height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Calculate crop quality (size * sharpness)
            area = (x2 - x1) * (y2 - y1)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality = area * sharpness

            # New Track
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "track_id": track_id,
                    "object_type": object_type,
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "frames_seen": 1,
                    "confirmed": False,
                    "best_quality": quality,
                    "best_crop": crop.copy(),
                    "best_bbox": [x1, y1, x2, y2],
                    "attributes": None,
                    "attributes_extracted": False,
                }
                print(f"[NEW TRACK] ID={track_id} Object={object_type}")

            # Existing Track
            else:
                track = self.active_tracks[track_id]
                track["last_seen"] = current_time
                track["frames_seen"] += 1

                # Confirm track
                if not track["confirmed"] and track["frames_seen"] >= self.min_confirm_frames:
                    track["confirmed"] = True
                    print(f"[TRACK CONFIRMED] ID={track_id} Object={track['object_type']} Frames={track['frames_seen']}")

                # Update best quality crop
                if quality > track["best_quality"]:
                    track["best_quality"] = quality
                    track["best_crop"] = crop.copy()
                    track["best_bbox"] = [x1, y1, x2, y2]

        return self._remove_missing_tracks(current_track_ids, current_time)

    def _remove_missing_tracks(self, current_track_ids, current_time):
        completed_tracks = []

        for track_id in list(self.active_tracks.keys()):
            track = self.active_tracks[track_id]

            if track_id not in current_track_ids:
                time_missing = current_time - track["last_seen"]

                if time_missing >= self.disappearance_timeout:
                    if track["confirmed"]:
                        completed_tracks.append(track)
                    else:
                        print(f"[IGNORED SHORT TRACK] ID={track_id} Object={track['object_type']} Frames={track['frames_seen']}")

                    print(f"[TRACK ENDED] ID={track_id} Object={track['object_type']} Frames={track['frames_seen']}")
                    del self.active_tracks[track_id]

        return completed_tracks

    def get_active_tracks(self):
        return self.active_tracks