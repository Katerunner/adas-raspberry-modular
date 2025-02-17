import time

import cv2
import numpy as np

from src.collision_warning.collision_warning_system import DEFAULT_DANGER_ZONE_COEFFICIENTS, CollisionWarningSystem
from src.modules.base_module import BaseModule
from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.speed_detection_module import SpeedDetectionModule

# Constants
ZONE_COLOR = (0, 165, 255)  # Orange color in BGR
PERIMETER_TRANSPARENCY = 0.3
IDLE_FILL_TRANSPARENCY = 0.1
TRIG_FILL_TRANSPARENCY = 0.3
PERIMETER_WIDTH = 1


class CollisionWarningModule(BaseModule):
    def __init__(
            self,
            object_detection_module: GeneralObjectDetectionModule,
            speed_detection_module: SpeedDetectionModule,
            frame_width: int,
            frame_height: int,
            danger_zone_coefficients: np.ndarray = DEFAULT_DANGER_ZONE_COEFFICIENTS,
            ttc: int = 2,
            confidence_tries: int = 2,
            triggered_lifetime: int = 3
    ):
        super().__init__()

        self.speed_detection_module = speed_detection_module
        self.object_detection_module = object_detection_module
        self.image_dims = np.array([frame_width, frame_height])
        self.collision_warning_system = CollisionWarningSystem(
            danger_zone_coefficients=danger_zone_coefficients,
            ttc=ttc,
            confidence_tries=confidence_tries,
            triggered_lifetime=triggered_lifetime
        )

    def perform(self):
        while not self._stop_event.is_set():
            time.sleep(0.01)
            registries = self.object_detection_module.value

            if (registries is None) or (registries.get("moving_object_registry") is None):
                continue

            moving_objects = registries.get("moving_object_registry").registry

            horizontal_shift = 0 if self.speed_detection_module.value is None else self.speed_detection_module.value[0]
            vertical_shift = 0 if self.speed_detection_module.value is None else self.speed_detection_module.value[1]
            self.collision_warning_system.update_state(
                image_dims=self.image_dims,
                moving_objects=moving_objects,
                horizontal_shift=horizontal_shift,
                vertical_shift=vertical_shift
            )
            self.value = self.collision_warning_system.triggered

    def draw_zone(self, frame: np.ndarray):
        if frame is None:
            return

        zone_coordinates = (self.collision_warning_system.danger_zone_coefficients * self.image_dims).astype(int)

        a = zone_coordinates[2].copy()
        zone_coordinates[2] = zone_coordinates[3].copy()
        zone_coordinates[3] = a

        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_coordinates], ZONE_COLOR)
        fill_transparency = TRIG_FILL_TRANSPARENCY if self.collision_warning_system.triggered else IDLE_FILL_TRANSPARENCY
        frame = cv2.addWeighted(overlay, fill_transparency, frame, 1 - fill_transparency, 0)
        cv2.polylines(frame, [zone_coordinates], isClosed=True, color=ZONE_COLOR, thickness=PERIMETER_WIDTH)
        return frame
