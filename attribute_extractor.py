import cv2
import numpy as np


class AttributeExtractor:

    def __init__(self):
        self.color_ranges = {
            "black": (0, 50),
            "gray": (50, 120),
            "white": (200, 255),
        }

    # =====================================================
    # MAIN EXTRACTION FUNCTION
    # =====================================================

    def extract(self, crop, object_type):
        """
        Extract attributes from the best crop of a tracked object.

        Person:
            shirt
            pants

        Vehicle:
            color
        """

        if crop is None or crop.size == 0:
            return {}

        # -------------------------------------------------
        # PERSON
        # -------------------------------------------------

        if object_type == "person":
            return self._extract_person(crop)

        # -------------------------------------------------
        # VEHICLES
        # -------------------------------------------------

        if object_type in ["car", "motorcycle", "bus", "truck"]:
            return {
                "color": self._estimate_color(crop)
            }

        # -------------------------------------------------
        # OTHER OBJECTS
        # -------------------------------------------------

        return {}

    # =====================================================
    # PERSON ATTRIBUTES
    # =====================================================

    def _extract_person(self, crop):

        height, width = crop.shape[:2]

        # Divide the person crop vertically.
        #
        # Upper ~45%  -> shirt
        # Lower ~45%  -> pants
        #
        # Ignore small regions near the very top/bottom.

        shirt_end = int(height * 0.45)

        pants_start = int(height * 0.50)

        shirt_crop = crop[
            int(height * 0.15):shirt_end,
            int(width * 0.15):int(width * 0.85)
        ]

        pants_crop = crop[
            pants_start:int(height * 0.90),
            int(width * 0.15):int(width * 0.85)
        ]

        shirt_color = self._estimate_color(shirt_crop)

        pants_color = self._estimate_color(pants_crop)

        # -------------------------------------------------
        # Backpack
        #
        # We are NOT guessing backpack presence using
        # random image heuristics yet.
        #
        # This will later be replaced by a proper detector.
        # -------------------------------------------------

        return {
            "shirt": shirt_color,
            "pants": pants_color,
        }

    # =====================================================
    # COLOR ESTIMATION
    # =====================================================

    def _estimate_color(self, crop):

        if crop is None or crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        pixels = hsv.reshape(-1, 3)

        h = pixels[:, 0]
        s = pixels[:, 1]
        v = pixels[:, 2]

        # -------------------------------------------------
        # BLACK
        # -------------------------------------------------

        black_mask = v < 50

        if np.mean(black_mask) > 0.35:
            return "black"

        # -------------------------------------------------
        # WHITE
        # -------------------------------------------------

        white_mask = (s < 40) & (v > 180)

        if np.mean(white_mask) > 0.35:
            return "white"

        # -------------------------------------------------
        # GRAY
        # -------------------------------------------------

        gray_mask = (s < 50) & (v >= 50) & (v <= 180)

        if np.mean(gray_mask) > 0.35:
            return "gray"

        # -------------------------------------------------
        # COLORED OBJECT
        # -------------------------------------------------

        valid = s > 50

        if np.sum(valid) == 0:
            return "unknown"

        dominant_hue = np.median(h[valid])

        # OpenCV hue range = 0-179

        if dominant_hue < 10 or dominant_hue >= 170:
            return "red"

        elif dominant_hue < 25:
            return "orange"

        elif dominant_hue < 35:
            return "yellow"

        elif dominant_hue < 85:
            return "green"

        elif dominant_hue < 130:
            return "blue"

        elif dominant_hue < 160:
            return "purple"

        else:
            return "red"