import cv2
import numpy as np

from src.forward_distance.forward_distance_calculator import ForwardDistanceCalculator
from src.modules.base_module import BaseModule
from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.image_reading_module import ImageReadingModule
from src.modules.perspective_transformation_module import PerspectiveTransformationModule


class ForwardDistanceModule(BaseModule):
    def __init__(
            self,
            image_reading_module: ImageReadingModule,
            object_detection_module: GeneralObjectDetectionModule,
            perspective_transformation_module: PerspectiveTransformationModule,
            zone_fraction=0.3,
            resolution=(256, 512)
    ):
        super().__init__()

        self.perspective_transformation_module = perspective_transformation_module
        self.object_detection_module = object_detection_module
        self.image_reading_module = image_reading_module
        self.zone_fraction = zone_fraction
        self.resolution = resolution

        self.forward_distance_calculator = ForwardDistanceCalculator(zone_fraction=self.zone_fraction)

    def draw_distance_text(self, image):
        """
        Draws the distance (stored in self.value) as text in the lower-right corner of the given image.
        """
        if image is None:
            return None
        h, w = image.shape[:2]
        if self.value is None:
            text = "Distance: N/A"
        else:
            text = f"Distance: {self.value:.1f}px"
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_w, text_h = text_size
        # Place text at lower-right with a 10px margin.
        x = w - text_w - 10
        y = h - 10
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        return image

    def draw_distance_visual(self, num_bars=5, active_color=(255, 0, 0), max_distance=480):
        """
        Creates a visual representation (an image) of the distance (from self.value) as a vertical bar meter.

        The output image has the specified resolution (default 256x512).
        The meter is built with 'num_bars' bars (default 5). The number of highlighted bars depends on the
        distance relative to max_distance:
          - if the distance is 0 (object is very close), all bars are highlighted,
          - if the distance is near max_distance, only the top bar is highlighted.
        If no distance is available (i.e. self.value is None), no bars are highlighted.

        The active (highlighted) bars are drawn in a bright, saturated blue (default),
        while inactive bars are drawn in a dull, dark blue.
        The bars are drawn as trapezoids, with the lower bars being wider and the top bars narrower.

        Returns:
            viz (np.ndarray): The visual representation image.
        """
        resolution = self.resolution
        viz = np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)  # white background
        width_v, height_v = resolution

        # Use self.value as the distance.
        if self.value is None:
            lit_bars = 0
        else:
            clamped = max(0, min(self.value, max_distance))
            lit_bars = int(round((1 - clamped / max_distance) * (num_bars - 1))) + 1
            # Force at least one active bar if a distance is available.
            lit_bars = max(1, min(num_bars, lit_bars))

        # Draw trapezoidal bars arranged vertically.
        gap = 5  # gap between bars
        total_gap = gap * (num_bars + 1)
        bar_height = (height_v - total_gap) / num_bars

        # Define widths: bottom bar is widest, top bar is narrowest.
        widest = width_v * 0.8
        narrowest = width_v * 0.2
        center_x = width_v // 2

        for i in range(num_bars):
            # i=0 is the bottom bar; i=num_bars-1 is the top bar.
            bar_bottom = height_v - gap - i * (bar_height + gap)
            bar_top = bar_bottom - bar_height

            # Linear interpolation for the widths.
            bottom_width = widest - (widest - narrowest) * (i / (num_bars - 1)) if num_bars > 1 else widest
            top_width = widest - (widest - narrowest) * (
                    (i + 1) / (num_bars - 1)) if num_bars > 1 and i < num_bars - 1 else bottom_width

            # Define trapezoid points: bottom-left, bottom-right, top-right, top-left.
            bl = (int(center_x - bottom_width / 2), int(bar_bottom))
            br = (int(center_x + bottom_width / 2), int(bar_bottom))
            tr = (int(center_x + top_width / 2), int(bar_top))
            tl = (int(center_x - top_width / 2), int(bar_top))
            pts = np.array([bl, br, tr, tl], np.int32).reshape((-1, 1, 2))

            # If self.value is None, lit_bars is 0 so the condition never holds.
            if lit_bars > 0 and i >= num_bars - lit_bars:
                color = active_color
            else:
                # Inactive: dull version (30% brightness).
                color = tuple(int(c * 0.3) for c in active_color)
            cv2.fillPoly(viz, [pts], color)

        return viz

    def perform(self):
        while not self._stop_event.is_set():
            if self.perspective_transformation_module.value is not None:
                input_image = self.perspective_transformation_module.value.copy()

                if self.object_detection_module.value:
                    centers = []
                    for obj in self.object_detection_module.value['moving_object_registry'].registry:
                        x1, y1, x2, y2 = obj.xyxy.astype(int)
                        centers.append(
                            self.perspective_transformation_module.transform_point((x1 + x2) // 2, y2)
                        )

                    self.value = self.forward_distance_calculator.calculate_distance(
                        image=input_image,
                        centers=centers
                    )
                else:
                    self.value = None
