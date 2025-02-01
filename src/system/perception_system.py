from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.image_reading_module import ImageReadingModule
from src.modules.lane_detection_module import LaneDetectionModule
from src.modules.perspective_transformation_module import PerspectiveTransformationModule
from src.modules.pothole_detection_module import PotholeDetectionModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule

from src.modules.collision_warning_module import CollisionWarningModule


class PerceptionSystem:
    def __init__(self,
                 image_reading_module: ImageReadingModule,
                 perspective_transformation_module: PerspectiveTransformationModule,
                 lane_detection_module: LaneDetectionModule,
                 sign_detection_module: TrafficSignDetectionModule,
                 general_object_detection_module: GeneralObjectDetectionModule,
                 pothole_detection_module: PotholeDetectionModule,
                 collision_warning_module: CollisionWarningModule):
        self.image_reading_module = image_reading_module
        self.perspective_transformation_module = perspective_transformation_module
        self.lane_detection_module = lane_detection_module
        self.sign_detection_module = sign_detection_module
        self.general_object_detection_module = general_object_detection_module
        self.pothole_detection_module = pothole_detection_module

        self.collision_warning_module = collision_warning_module

    def start(self):
        self.image_reading_module.start()
        self.perspective_transformation_module.start()
        self.lane_detection_module.start()
        self.sign_detection_module.start()
        self.general_object_detection_module.start()
        self.pothole_detection_module.start()
        self.collision_warning_module.start()

    def stop(self):
        self.image_reading_module.stop()
        self.perspective_transformation_module.stop()
        self.lane_detection_module.stop()
        self.sign_detection_module.stop()
        self.general_object_detection_module.stop()
        self.pothole_detection_module.stop()
        self.collision_warning_module.stop()

    @property
    def frame(self,
              draw_collision_zone: bool = True,
              draw_perspective_guidelines: bool = True,
              draw_moving_objects: bool = True,
              draw_lanes: bool = True,
              ):
        frame = self.image_reading_module.value
        frame = self.collision_warning_module.draw_zone(frame) if draw_collision_zone else frame
        frame = self.lane_detection_module.draw_lanes(frame) if draw_lanes else frame
        frame = self.general_object_detection_module.draw_moving_objects(frame) if draw_moving_objects else frame
        frame = self.perspective_transformation_module.draw_guidelines(frame) if draw_perspective_guidelines else frame
        return frame

    @property
    def perspective_frame(self):
        return self.perspective_transformation_module.value

    @property
    def moving_object_registry(self):
        if self.general_object_detection_module.value is not None:
            return self.general_object_detection_module.value.get("moving_object_registry")

    @property
    def traffic_light_registry(self):
        if self.general_object_detection_module.value is not None:
            return self.general_object_detection_module.value.get("traffic_light_registry")

    @property
    def traffic_lane_registry(self):
        return self.lane_detection_module.value

    @property
    def traffic_sign_registry(self):
        return self.sign_detection_module.value

    @property
    def pothole_registry(self):
        return self.pothole_detection_module.value
