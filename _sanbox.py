import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from src.lane_detection.lane_processor import LaneProcessor
from src.modules.image_reading_module import ImageReadingModule
from src.modules.lane_detection_module import LaneDetectionModule

video_path = "assets/videos/video_1.mp4"
lane_model_path = "trained_models/lane-yolov8n.pt"

lane_yolo = YOLO(lane_model_path)

colors = {
    0: (255, 0, 0),  # LL: Blue
    1: (0, 255, 0),  # LC: Green
    2: (0, 0, 255),  # RC: Red
    3: (255, 255, 0)  # RR: Cyan
}

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
lane_detection_module = LaneDetectionModule(source_module=image_reading_module, model_weights=lane_model_path)

image_reading_module.start()
lane_detection_module.start()

while True:
    lane_processor = lane_detection_module.value
    frame = image_reading_module.value

    if lane_processor:
        for lane_id in lane_processor.lane_labels:
            lane_cls = lane_processor.lane_labels[lane_id]
            lane_obj = lane_processor.lanes[lane_id]
            if lane_obj:

                for p, conf in zip(lane_obj.points, lane_obj.confs):
                    text = f"{lane_processor.lane_labels[lane_id]} Conf: {round(conf, 2)}"
                    cv2.circle(frame, p, 5, colors[lane_id], thickness=-1)
                    cv2.putText(
                        frame,
                        text,
                        (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                cv2.polylines(frame, [lane_obj.estimated_points], isClosed=False, color=colors[lane_id], thickness=2)

    if frame is not None:
        cv2.imshow('Lane Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        image_reading_module.stop()
        lane_detection_module.stop()
        break

cv2.destroyAllWindows()

#
# cap = cv2.VideoCapture(video_path)
#
# colors = {
#     0: (255, 0, 0),  # LL: Blue
#     1: (0, 255, 0),  # LC: Green
#     2: (0, 0, 255),  # RC: Red
#     3: (255, 255, 0)  # RR: Cyan
# }
#
# confidence_threshold = 0.3
#
# if not cap.isOpened():
#     print("Error opening video file")
#
#
# def predict_lane_points(points):
#     if len(points) < 3:
#         return []
#     x_orig = np.array([p[0] for p in points])
#     y_orig = np.array([p[1] for p in points])
#     y_min = np.min(y_orig)
#     y_max = np.max(y_orig)
#     model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(min_samples=2, loss="squared_error"))
#     model.fit(y_orig.reshape(-1, 1), x_orig)
#     y_pred = np.linspace(y_min, y_max, 10)
#     X_pred = model.predict(y_pred.reshape(-1, 1))
#     return [(int(x), int(y)) for x, y in zip(X_pred, y_pred)]
#
#
# lane_labels = {0: "LL", 1: "LC", 2: "RC", 3: "RR"}
#
# frame_counter = 0
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_counter += 1
#     if frame_counter % 2 == 0:
#         continue  # Skip every 2nd frame
#
#     lane_results = lane_yolo.predict(frame, verbose=False)
#     # lane_boxes = lane_results[0].boxes
#     # lane_classes = lane_boxes.cls
#     # lane_confs = lane_boxes.conf
#     # lane_xywhns = lane_boxes.xywhn
#     #
#     # lane_points = {0: [], 1: [], 2: [], 3: []}
#     #
#     # for i in range(len(lane_classes)):
#     #     conf = lane_confs[i].item()
#     #     if conf >= confidence_threshold:
#     #         lane_class = int(lane_classes[i].item())
#     #         if lane_class in lane_points:
#     #             cx = lane_xywhns[i][0].item() * frame.shape[1]
#     #             cy = lane_xywhns[i][1].item() * frame.shape[0]
#     #             lane_points[lane_class].append((int(cx), int(cy)))
#
#     lane_processor = LaneProcessor.from_yolo_result(
#         yolo_result=lane_results,
#         frame_shape=frame.shape[:2],
#         confidence_threshold=0.2,
#     )
#     lane_processor.estimate_lane_curves(use_weights=True)
#
#     for lane_id in lane_processor.lane_labels:
#         lane_cls = lane_processor.lane_labels[lane_id]
#         lane_obj = lane_processor.lanes[lane_id]
#         if lane_obj:
#
#             for p, conf in zip(lane_obj.points, lane_obj.confs):
#                 text = f"{lane_labels[lane_id]} Conf: {round(conf, 2)}"
#                 cv2.circle(frame, p, 5, colors[lane_id], thickness=-1)
#                 cv2.putText(
#                     frame,
#                     text,
#                     (p[0] + 5, p[1] - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     1
#                 )
#
#             cv2.polylines(frame, [lane_obj.estimated_points], isClosed=False, color=colors[lane_id], thickness=2)
#
#     cv2.imshow('Lane Detection', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
