import cv2
from ultralytics import YOLO

from src.pools.base_pool import BasePool
from src.modules.image_reading_module import ImageReadingModule
from src.modules.lane_detection_module import LaneDetectionModule
from src.lane_detection.lane_curve_estimator import LaneCurveEstimator
from src.lane_detection.lane_processor_corrector import LaneProcessorCorrector

video_path = "assets/videos/video_1.mp4"
lane_model_path = "trained_models/lane-yolov8n.pt"

lane_yolo = YOLO(lane_model_path, task='predict')

colors = {
    0: (255, 0, 0),  # LL: Blue
    1: (0, 255, 0),  # LC: Green
    2: (0, 0, 255),  # RC: Red
    3: (255, 255, 0)  # RR: Cyan
}

lane_curve_estimator = LaneCurveEstimator(image_shape=(1, 1))
lane_processor_corrector = LaneProcessorCorrector(lane_overlap=10, y_tolerance=5)

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
lane_detection_modules = [
    LaneDetectionModule(
        source_module=image_reading_module,
        model_weights=lane_model_path,
        lane_curve_estimator=lane_curve_estimator,
        lane_processor_corrector=lane_processor_corrector
    ),
    LaneDetectionModule(
        source_module=image_reading_module,
        model_weights=lane_model_path,
        lane_curve_estimator=lane_curve_estimator,
        lane_processor_corrector=lane_processor_corrector
    )
]
lane_detection_pool = BasePool(result_format='last',
                               workers=lane_detection_modules,
                               intermodule_start_interval=1 / 30 / len(lane_detection_modules),
                               intermodule_read_interval=1 / 30 / len(lane_detection_modules))

# Start the modules
image_reading_module.start()
lane_detection_pool.start()

while True:
    lane_registry = lane_detection_pool.value
    frame = image_reading_module.value

    if lane_registry:
        for lane_id in lane_registry.lane_labels:
            lane_cls = lane_registry.lane_labels[lane_id]
            lane_obj = lane_registry.lanes[lane_id]
            if lane_obj and len(lane_obj.estimated_points) > 0:
                cv2.polylines(frame, [lane_obj.estimated_points], isClosed=False, color=colors[lane_id], thickness=2)

    if frame is not None:
        cv2.imshow('Lane Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        image_reading_module.stop()
        lane_detection_pool.stop()
        break

cv2.destroyAllWindows()
