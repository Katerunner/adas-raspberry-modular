import time

import cv2
import numpy as np

from src.moving_object_detection.moving_object import MovingObject

DEFAULT_DANGER_ZONE_COEFFICIENTS = np.float32([
    [0.2, 1.0],  # Bottom-left
    [0.8, 1.0],  # Bottom-right
    [0.4, 0.75],  # Top-left (near center)
    [0.6, 0.75]  # Top-right (near center)
])


class CollisionWarningSystem:
    def __init__(self,
                 danger_zone_coefficients: np.ndarray = DEFAULT_DANGER_ZONE_COEFFICIENTS,
                 ttc: int = 2,
                 confidence_tries: int = 2,
                 triggered_lifetime: int = 3):
        self.danger_zone_coefficients = danger_zone_coefficients.copy()
        self.top_l_x = danger_zone_coefficients[2][0].copy()
        self.top_r_x = danger_zone_coefficients[3][0].copy()
        self.top_l_y = danger_zone_coefficients[2][1].copy()
        self.top_r_y = danger_zone_coefficients[3][1].copy()
        self.ttc = ttc
        self.triggered_lifetime = triggered_lifetime
        self.confidence_tries = confidence_tries
        self._triggered = False
        self._triggered_until = 0
        self._danger_registry = {}

    @property
    def triggered(self) -> bool:
        if self._triggered and time.time() > self._triggered_until:
            self._triggered = False
        return self._triggered

    def update_state(self, image_dims: tuple[float, float],
                     moving_objects: list[MovingObject],
                     horizontal_shift: float = 0.0,
                     vertical_shift: float = 0.0):

        self.danger_zone_coefficients[2][0] = self.top_l_x - horizontal_shift / 1000
        self.danger_zone_coefficients[3][0] = self.top_r_x - horizontal_shift / 1000
        self.danger_zone_coefficients[2][1] = (self.top_l_y + np.abs(horizontal_shift / 4000)) * (
                1 - np.sqrt(np.abs(vertical_shift / 10000)))
        self.danger_zone_coefficients[3][1] = (self.top_r_y + np.abs(horizontal_shift / 4000)) * (
                1 - np.sqrt(np.abs(vertical_shift / 10000)))

        zone_coordinates = self.danger_zone_coefficients * np.array(image_dims)
        for obj in moving_objects:
            x_now, y_now = np.mean([obj.xyxy[0], obj.xyxy[2]]), obj.xyxy[3]
            x_aft, y_aft = obj.predict_position(s_after=self.ttc)

            # Generate intermediate points along the line
            num_points = 50  # Number of points to sample along the line
            x_points = np.linspace(x_now, x_aft, num_points)
            y_points = np.linspace(y_now, y_aft, num_points)

            # Check if any point along the line is inside the danger zone
            for x, y in zip(x_points, y_points):
                if self._is_inside_danger_zone(x, y, zone_coordinates):
                    self._danger_registry[obj.name] = self._danger_registry.get(obj.name, 0) + 1
                    if self._danger_registry[obj.name] >= self.confidence_tries:
                        self._triggered = True
                        self._triggered_until = time.time() + self.triggered_lifetime
                    break
            else:
                # Remove from registry if no points are inside the danger zone
                if obj.name in self._danger_registry:
                    del self._danger_registry[obj.name]

    @staticmethod
    def _is_inside_danger_zone(x, y, zone_coordinates):
        # Convert zone coordinates to integer and create a polygon
        polygon = zone_coordinates.astype(np.int32)

        # Use cv2.pointPolygonTest to check if the point is inside the polygon
        point = (int(x), int(y))
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0  # Returns True if inside or on the edge
