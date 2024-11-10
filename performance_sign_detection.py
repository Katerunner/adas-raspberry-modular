import cv2
import numpy as np

# from src.pools.base_pool import BasePool
from src.modules.image_reading_module import ImageReadingModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule

video_path = "assets/videos/video_1.mp4"
sign_model_path = "trained_models/sign-yolov8n.pt"

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
sign_detection_module = TrafficSignDetectionModule(source_module=image_reading_module, model_weights=sign_model_path)

# Start the modules
image_reading_module.start()
sign_detection_module.start()

while True:
    sign_registry = sign_detection_module.value
    frame = image_reading_module.value

    if sign_registry and frame is not None:
        frame = frame.copy()
        for sign in sign_registry.registry:
            x1, y1, x2, y2 = sign.position
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID: {sign.guid} | Name: {sign.name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the annotated frame
    if frame is not None:
        cv2.imshow('Traffic Sign Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        image_reading_module.stop()
        sign_detection_module.stop()
        break

cv2.destroyAllWindows()
