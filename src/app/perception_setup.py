from src.lane_detection.lane_curve_estimator import LaneCurveEstimator
from src.lane_detection.lane_processor_corrector import LaneProcessorCorrector
from src.modules.collision_warning_module import CollisionWarningModule
from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.image_reading_module import ImageReadingModule
from src.modules.lane_detection_module import LaneDetectionModule
from src.modules.perspective_transformation_module import PerspectiveTransformationModule
from src.modules.pothole_detection_module import PotholeDetectionModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule
from src.system.perception_system import PerceptionSystem


def setup_perception_system():
    video_path = "assets/videos/video_2.mp4"
    object_model_path = "trained_models/moob-yolov8n.pt"
    lane_model_path = "trained_models/lane-yolov8n.pt"
    sign_model_path = "trained_models/sign-yolov8n.pt"
    hole_model_path = "trained_models/hole-yolov8n.pt"

    image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
    object_detection_module = GeneralObjectDetectionModule(
        source_module=image_reading_module,
        model_weights=object_model_path,
    )
    lane_curve_estimator = LaneCurveEstimator(memory_size=25)
    lane_processor_corrector = LaneProcessorCorrector(lane_overlap=10, y_tolerance=5)
    lane_detection_module = LaneDetectionModule(
        source_module=image_reading_module,
        model_weights=lane_model_path,
        lane_curve_estimator=lane_curve_estimator,
        lane_processor_corrector=lane_processor_corrector
    )
    sign_detection_module = TrafficSignDetectionModule(
        source_module=image_reading_module,
        model_weights=sign_model_path
    )
    perspective_transformation_module = PerspectiveTransformationModule(source_module=image_reading_module)
    pothole_detection_module = PotholeDetectionModule(
        source_module=image_reading_module,
        model_weights=hole_model_path
    )
    collision_warning_module = CollisionWarningModule(
        object_detection_module=object_detection_module,
        frame_width=image_reading_module.frame_width,
        frame_height=image_reading_module.frame_height
    )
    ps = PerceptionSystem(
        image_reading_module=image_reading_module,
        perspective_transformation_module=perspective_transformation_module,
        lane_detection_module=lane_detection_module,
        sign_detection_module=sign_detection_module,
        general_object_detection_module=object_detection_module,
        pothole_detection_module=pothole_detection_module,
        collision_warning_module=collision_warning_module
    )
    return ps
