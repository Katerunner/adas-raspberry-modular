import cv2
from ultralytics import YOLO

from src.object_tracking.naive_object_tracker import NaiveObjectTracker
from src.modules.image_reading_module import ImageReadingModule

video_path = "assets/videos/video_1.mp4"
sign_model_path = "trained_models/sign-yolov8n.pt"

sign_yolo = YOLO(sign_model_path, task='track')

# Colors for different lanes (adjust as needed)
colors = {
    0: (255, 0, 0),  # LL: Blue
    1: (0, 255, 0),  # LC: Green
    2: (0, 0, 255),  # RC: Red
    3: (255, 255, 0)  # RR: Cyan
}

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
object_tracker = NaiveObjectTracker(max_objects=20,
                                    feature_threshold=0.875,
                                    position_threshold=0.95,
                                    lifespan=2)

# Start the image reading module
image_reading_module.start()

while True:
    frame = image_reading_module.value

    if frame is not None:
        # Get predictions from YOLO model
        prediction_result = sign_yolo.predict(frame, conf=0.7, verbose=False)[0]
        original_image = prediction_result.orig_img

        # Extract bounding boxes and confidences
        boxes = prediction_result.boxes
        confs = boxes.conf.numpy() if boxes.conf is not None else None
        xyxys = boxes.xyxy.numpy() if boxes.xyxy is not None else None

        # Process the results to get tracking IDs
        ids = object_tracker.process_yolo_result(prediction_result=prediction_result)

        # Draw bounding boxes, confidence scores, and tracker IDs
        if xyxys is not None and confs is not None:
            for i, (xyxy, conf) in enumerate(zip(xyxys, confs)):
                x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to integers
                tracker_id = ids[i]  # Get tracker ID from the custom tracker

                # Choose a color based on the lane type if applicable (default to white if not specified)
                color = colors.get(tracker_id % len(colors), (255, 255, 255))

                # Draw the bounding box
                cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)

                # Display confidence and tracker ID
                label = f"ID: {tracker_id} | Conf: {conf:.2f}"
                cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display the image
        cv2.imshow('Lane Detection', original_image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        image_reading_module.stop()
        break

# Cleanup
cv2.destroyAllWindows()
