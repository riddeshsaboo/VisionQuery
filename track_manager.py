import time
import cv2


class TrackManager:
    def __init__(self, disappearance_timeout=2.0, min_confirm_frames=3):
        self.disappearance_timeout = disappearance_timeout
        self.min_confirm_frames = min_confirm_frames
        self.active_tracks = {}

    def update(self, results, frame, media_time=None):
        current_time = time.time()
        if media_time is None:
            media_time = current_time

        current_track_ids = set()
        if not results:
            return []

        result = results[0]
        if result.boxes.id is None:
            return self._remove_missing_tracks(current_track_ids, current_time)

        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xyxy.int().cpu().tolist()
        frame_height, frame_width = frame.shape[:2]

        for track_id, class_id, box in zip(track_ids, classes, boxes):
            track_id, class_id = int(track_id), int(class_id)
            current_track_ids.add(track_id)

            bbox = self._clamp_bbox(box, frame_width, frame_height)
            if bbox is None:
                continue

            crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.size == 0:
                continue

            quality = self._calculate_crop_quality(crop)
            if track_id not in self.active_tracks:
                self._create_track(track_id, result.names[class_id], current_time, media_time, quality, crop, bbox)
            else:
                self._update_track(track_id, current_time, media_time, quality, crop, bbox)

        return self._remove_missing_tracks(current_track_ids, current_time)

    def _clamp_bbox(self, box, frame_width, frame_height):
        x1 = max(0, min(box[0], frame_width - 1))
        y1 = max(0, min(box[1], frame_height - 1))
        x2 = max(0, min(box[2], frame_width - 1))
        y2 = max(0, min(box[3], frame_height - 1))
        return [x1, y1, x2, y2] if (x2 > x1 and y2 > y1) else None

    def _calculate_crop_quality(self, crop):
        area = crop.shape[0] * crop.shape[1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return area * sharpness

    def _create_track(self, track_id, object_type, current_time, media_time, quality, crop, bbox):
        self.active_tracks[track_id] = {
            "track_id": track_id,
            "object_type": object_type,
            "first_seen": current_time,
            "last_seen": current_time,
            "first_media_time": media_time,
            "last_media_time": media_time,
            "frames_seen": 1,
            "confirmed": False,
            "best_quality": quality,
            "best_crop": crop.copy(),
            "best_bbox": bbox,
            "attribute_crops": [(quality, crop.copy())],
        }
        print(f"[NEW TRACK] ID={track_id} Object={object_type}")

    def _update_track(self, track_id, current_time, media_time, quality, crop, bbox):
        track = self.active_tracks[track_id]
        track["last_seen"] = current_time
        track["last_media_time"] = media_time
        track["frames_seen"] += 1

        if not track["confirmed"] and track["frames_seen"] >= self.min_confirm_frames:
            track["confirmed"] = True
            print(f"[TRACK CONFIRMED] ID={track_id} Object={track['object_type']} Frames={track['frames_seen']}")

        if quality > track["best_quality"]:
            track["best_quality"] = quality
            track["best_crop"] = crop.copy()
            track["best_bbox"] = bbox

        # Keep the 5 highest-quality crops for attribute extraction
        track["attribute_crops"].append((quality, crop.copy()))
        track["attribute_crops"].sort(key=lambda x: x[0], reverse=True)
        track["attribute_crops"] = track["attribute_crops"][:10]

    def _remove_missing_tracks(self, current_track_ids, current_time):
        completed_tracks = []
        for track_id in list(self.active_tracks):
            if track_id in current_track_ids:
                continue

            track = self.active_tracks[track_id]
            if (current_time - track["last_seen"]) < self.disappearance_timeout:
                continue

            if track["confirmed"]:
                completed_tracks.append(track)
            else:
                print(f"[IGNORED SHORT TRACK] ID={track_id} Object={track['object_type']} Frames={track['frames_seen']}")

            print(f"[TRACK ENDED] ID={track_id} Object={track['object_type']} Frames={track['frames_seen']}")
            del self.active_tracks[track_id]

        return completed_tracks

    def get_active_tracks(self):
        return self.active_tracks