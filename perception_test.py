import cv2

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

video_path = "assets/videos/video_2.mp4"
object_model_path = "trained_models/moob-yolov8n.pt"
lane_model_path = "trained_models/lane-yolov8n.pt"
sign_model_path = "trained_models/sign-yolov8n.pt"
hole_model_path = 'trained_models/hole-yolov8n.pt'

lane_colors = {
    0: (255, 0, 0),  # LL: Blue
    1: (0, 255, 0),  # LC: Green
    2: (0, 0, 255),  # RC: Red
    3: (255, 255, 0)  # RR: Cyan
}

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
pothole_detection_module = PotholeDetectionModule(source_module=image_reading_module, model_weights=hole_model_path)

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

ps.start()

while True:
    frame = ps.frame

    if ps.traffic_sign_registry and frame is not None:
        for sign in ps.traffic_sign_registry.registry:
            x1, y1, x2, y2 = sign.position
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID: {sign.guid} | Name: {sign.name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if ps.traffic_lane_registry:
        for lane_id in ps.traffic_lane_registry.lane_labels:
            lane_cls = ps.traffic_lane_registry.lane_labels[lane_id]
            lane_obj = ps.traffic_lane_registry.lanes[lane_id]
            if lane_obj and len(lane_obj.estimated_points) > 0:
                cv2.polylines(
                    frame,
                    [lane_obj.estimated_points],
                    isClosed=False,
                    color=lane_colors[lane_id],
                    thickness=2
                )

    if ps.moving_object_registry and frame is not None:
        for obj in ps.moving_object_registry.registry:
            x1, y1, x2, y2 = obj.xyxy.astype(int)
            cv2.line(frame, (x1, y2), (x2, y2), (0, 255, 0), 2)

            label = f"ID: {obj.guid} | Name: {obj.name}"
            cv2.putText(frame, label, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            x_center = (x1 + x2) // 2
            y_bottom = int(y2)
            pr_result = obj.predict_position(s_after=2)
            x_a, y_a = pr_result
            if x_a is not None:
                cv2.line(frame, (x_center, y_bottom), (int(x_a), int(y_a)), (0, 255, 0), 2)

    if ps.traffic_light_registry and frame is not None:
        for obj in ps.traffic_light_registry.registry:
            x1, y1, x2, y2 = obj.xyxy.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Color: {obj.color}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if ps.pothole_registry and frame is not None:
        for pot in ps.pothole_registry:
            p_x1, p_y1, p_x2, p_y2 = pot.xyxy.astype(int)
            cv2.line(frame, (p_x1, p_y2), (p_x2, p_y2), (0, 0, 255), 3)

            label = f"pothole {pot.conf:.2f}"
            cv2.putText(frame, label, (p_x1, p_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the annotated frame
    if frame is not None:
        cv2.imshow('Processed Detection', frame)
        cv2.imshow('Perspective Detection', ps.perspective_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ps.stop()
        break

cv2.destroyAllWindows()
