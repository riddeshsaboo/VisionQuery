import cv2
import numpy as np
import os 

class AttributeExtractor:
    def __init__(self):
        self.colors = [
            "black", "white", "gray", "red", "orange",
            "yellow", "green", "blue", "purple", "brown"
        ]

        self.debug_enabled = True
        self.debug_count = 0

    os.makedirs("debug_crops/shirts", exist_ok=True)
    os.makedirs("debug_crops/pants", exist_ok=True)

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

        # Inset horizontally to exclude surrounding background
        x1, x2 = int(w * 0.25), int(w * 0.75)
        shirt_crop = crop[int(h * 0.20):int(h * 0.45), x1:x2]
        pants_crop = crop[int(h * 0.48):int(h * 0.70), x1:x2]

        return {
            "shirt": self._estimate_person_color(shirt_crop),
            "pants": self._estimate_person_color(pants_crop),
        }

    def _estimate_person_color(self, crop):
        if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return "unknown"

        h, w = crop.shape[:2]

        # Ignore edges where background is most likely to appear
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        y1 = int(h * 0.15)
        y2 = int(h * 0.85)

        crop = crop[y1:y2, x1:x2]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
        total = H.size
        if total < 10:
            return "unknown"

        # 1. Achromatic scores (hue ignored to prevent desaturated tones mapping to blue)
        low_sat = S < 50
        scores = {
            "black": np.mean(low_sat & (V < 105)),
            "dark_gray": np.mean(low_sat & (V >= 105) & (V < 145)),
            "gray": np.mean(low_sat & (V >= 145) & (V < 195)),
            "white": np.mean(low_sat & (V >= 195)),
        }

        # 2. Chromatic scores weighted by saturation and brightness
        colorful = S >= 50
        if np.sum(colorful) > 5:
            h, s, v = H[colorful], S[colorful], V[colorful]
            weights = (s / 255.0) * (v / 255.0)

            def ratio(mask):
                return float(np.sum(weights[mask]) / total) if np.any(mask) else 0.0

            red = ((h < 10) | (h >= 170)) & (s >= 70)
            green = (h >= 38) & (h < 85) & (s >= 60)
            blue = (h >= 85) & (h < 135) & (s >= 60)

            scores.update({
                "red": ratio(red),
                "dark_red": ratio(red & (v < 100)),
                "orange": ratio((h >= 10) & (h < 22) & (s >= 70) & (v >= 80)),
                "brown": ratio((h >= 5) & (h < 25) & (s >= 45) & (s < 180) & (v < 150)),
                "yellow": ratio((h >= 22) & (h < 38) & (s >= 70) & (v >= 90)),
                "beige": ratio((h >= 15) & (h < 40) & (s >= 30) & (s < 110) & (v >= 130)),
                "green": ratio(green),
                "dark_green": ratio(green & (v < 100)),
                "blue": ratio(blue),
                "dark_blue": ratio(blue & (v < 100)),
                "light_blue": ratio(blue & (v > 175)),
                "purple": ratio((h >= 135) & (h < 165) & (s >= 60)),
                "pink": ratio(((h >= 155) | (h < 175)) & (s >= 45) & (v >= 120)),
            })

        best_color = max(scores, key=scores.get)
        return best_color if scores[best_color] >= 0.04 else "unknown"

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


    def extract_multiple(self, crops, object_type):
        if not crops:
            return {}

        # Unpack nested tuples to extract raw image crops
        actual_crops = []
        for item in crops:
            crop = item
            while isinstance(crop, tuple):
                crop = crop[-1]
            actual_crops.append(crop)

        if object_type != "person":
            return self.extract(actual_crops[0], object_type)

        shirts, pants = [], []
        for crop in actual_crops:
            attrs = self._extract_person(crop)
            s, p = attrs.get("shirt"), attrs.get("pants")
            if s and s != "unknown": shirts.append(s)
            if p and p != "unknown": pants.append(p)

        def majority(colors):
            return max(set(colors), key=colors.count) if colors else "unknown"

        return {"shirt": majority(shirts), "pants": majority(pants)}