import cv2
import numpy as np


class AttributeExtractor:
    def __init__(self):
        self.colors = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "brown"]

    def extract(self, crop, object_type):
        if crop is None or crop.size == 0:
            return {}
        if object_type == "person":
            return self._extract_person(crop)
        if object_type in ["car", "motorcycle", "bus", "truck"]:
            return {"color": self._estimate_color(crop)}
        return {}

    def _extract_person(self, crop):
        h, w = crop.shape[:2]
        if h < 30 or w < 15:
            return {"shirt": "unknown", "pants": "unknown"}

        # Inset horizontal boundaries to avoid background
        x1, x2 = int(w * 0.15), int(w * 0.85)

        # Torso (shirt) and lower body (pants) regions
        shirt_crop = crop[int(h * 0.25):int(h * 0.52), x1:x2]
        pants_crop = crop[int(h * 0.55):int(h * 0.88), x1:x2]

        return {
            "shirt": self._estimate_color(shirt_crop),
            "pants": self._estimate_color(pants_crop),
        }

    def _estimate_color(self, crop):
        if crop is None or crop.size == 0:
            return "unknown"

        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return "unknown"

        # Crop edges to reduce boundary noise
        bx, by = max(1, int(w * 0.08)), max(1, int(h * 0.08))
        crop = crop[by:h - by, bx:w - bx]
        if crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_channel, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Achromatic & brown checks
        if np.mean((v < 75) & (s < 90)) > 0.35:
            return "black"
        if np.mean((s < 45) & (v > 175)) > 0.40:
            return "white"
        if np.mean((s < 45) & (v >= 75) & (v <= 175)) > 0.40:
            return "gray"
        if np.mean((h_channel >= 5) & (h_channel < 25) & (s > 50) & (v < 150)) > 0.25:
            return "brown"

        # Dominant hue detection for chromatic colors
        color_mask = (s > 55) & (v > 45)
        if np.sum(color_mask) < 10:
            return "unknown"

        hist = cv2.calcHist([h_channel.astype(np.uint8)], [0], color_mask.astype(np.uint8), [180], [0, 180])
        hue = int(np.argmax(hist))

        # Hue mapping
        if hue < 10 or hue >= 170:
            return "red"
        elif hue < 20:
            return "orange"
        elif hue < 38:
            return "yellow"
        elif hue < 85:
            return "green"
        elif hue < 135:
            return "blue"
        elif hue < 165:
            return "purple"
        return "red"