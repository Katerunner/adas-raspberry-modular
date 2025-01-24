from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.image_reading_module import ImageReadingModule
from src.modules.lane_detection_module import LaneDetectionModule
from src.modules.perspective_transformation_module import PerspectiveTransformationModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule


class PerceptionSystem:
    def __init__(self,
                 image_reading_module: ImageReadingModule,
                 perspective_transformation_module: PerspectiveTransformationModule,
                 lane_detection_module: LaneDetectionModule,
                 sign_detection_module: TrafficSignDetectionModule,
                 general_object_detection_module: GeneralObjectDetectionModule):
        self.image_reading_module = image_reading_module
        self.perspective_transformation_module = perspective_transformation_module
        self.lane_detection_module = lane_detection_module
        self.sign_detection_module = sign_detection_module
        self.general_object_detection_module = general_object_detection_module

    def start(self):
        self.image_reading_module.start()
        self.perspective_transformation_module.start()
        self.lane_detection_module.start()
        self.sign_detection_module.start()
        self.general_object_detection_module.start()

    def stop(self):
        self.image_reading_module.stop()
        self.perspective_transformation_module.stop()
        self.lane_detection_module.stop()
        self.sign_detection_module.stop()
        self.general_object_detection_module.stop()

    @property
    def frame(self):
        return self.image_reading_module.value

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
