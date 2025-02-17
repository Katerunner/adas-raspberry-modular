import time

import cv2
import numpy as np

from src.lane_detection.lane_registry import LaneRegistry
from src.led_strip.led_strip_processor import LEDStripProcessor
from src.modules.base_module import BaseModule
from src.modules.collision_warning_module import CollisionWarningModule
from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.lane_detection_module import LaneDetectionModule
from src.modules.perspective_transformation_module import PerspectiveTransformationModule
from src.modules.pothole_detection_module import PotholeDetectionModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule


class LEDStripModule(BaseModule):
    def __init__(self,
                 perspective_transformation_module: PerspectiveTransformationModule,
                 lane_detection_module: LaneDetectionModule,
                 pothole_detection_module: PotholeDetectionModule,
                 collision_warning_module: CollisionWarningModule,
                 object_detection_module: GeneralObjectDetectionModule,
                 traffic_sign_detection_module: TrafficSignDetectionModule,
                 width: int = 640, height: int = 80):
        super().__init__()
        self.traffic_sign_detection_module = traffic_sign_detection_module
        self.perspective_transformation_module = perspective_transformation_module
        self.lane_detection_module = lane_detection_module
        self.pothole_detection_module = pothole_detection_module
        self.collision_warning_module = collision_warning_module
        self.object_detection_module = object_detection_module

        self.led_strip_processor = LEDStripProcessor(width=width, height=height)

    def perform(self):
        """Perform traffic sign detection on frames from the source module."""
        while not self._stop_event.is_set():
            self.value = self.led_strip_processor.update()
            width = self.perspective_transformation_module.dst_width

            if self.lane_detection_module.value:
                lane_registry: LaneRegistry = self.lane_detection_module.value

                for lane in lane_registry.lanes:
                    if lane:
                        points_to_consider = sorted(lane.estimated_points, key=lambda c: c[1], reverse=True)
                        for point in points_to_consider[:3]:
                            x, y = self.perspective_transformation_module.transform_point(point[0], point[1])
                            rx = x / width
                            self.led_strip_processor.add_event(
                                relative_x=np.clip(rx, 0.0, 1.0),
                                width=0.2,
                                intensity=0.2,
                                color=(255, 255, 255),
                            )

            if self.pothole_detection_module.value:
                potholes = self.pothole_detection_module.value
                for pothole in potholes:
                    x_center = (pothole.xyxy[0] + pothole.xyxy[2]) / 2
                    y_bottom = pothole.xyxy[3]
                    x, y = self.perspective_transformation_module.transform_point(x_center, y_bottom)
                    rx = x / width
                    self.led_strip_processor.add_event(
                        relative_x=np.clip(rx, 0.0, 1.0),
                        width=0.5,
                        intensity=0.05,
                        color=(0, 0, 255),
                    )

            if self.collision_warning_module.value:
                self.led_strip_processor.add_event(
                    relative_x=0.5,
                    width=15.0,
                    intensity=0.75,
                    lifetime=0.5,
                    color=(0, 165, 255),
                )

            if self.object_detection_module.value:
                traffic_lights = self.object_detection_module.value.get("traffic_light_registry").registry
                traffic_lights_data = []
                for traffic_light in traffic_lights:
                    color = traffic_light.color
                    xyxy = traffic_light.xyxy
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    width_n = self.object_detection_module.source_module.frame_width
                    relative_x = center_x / width_n
                    traffic_lights_data.append((relative_x, color, 0.1))

                self.led_strip_processor.add_traffic_light(traffic_lights_data)

            if self.traffic_sign_detection_module.value:
                traffic_sign_registry = self.traffic_sign_detection_module.value.registry
                for traffic_sign in traffic_sign_registry:
                    if traffic_sign.name == 'information--pedestrians-crossing':
                        self.led_strip_processor.add_pedestrian_warning(lifetime=0.1)
