import cv2

from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.image_reading_module import ImageReadingModule

video_path = "assets/videos/video_2.mp4"
object_model_path = "trained_models/moob-yolov8n.pt"

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
object_detection_module = GeneralObjectDetectionModule(
    source_module=image_reading_module,
    model_weights=object_model_path
)

# Start the modules
image_reading_module.start()
object_detection_module.start()

while True:
    object_registry = object_detection_module.value
    object_registry = object_registry.get("moving_object_registry") if object_registry is not None else None
    frame = image_reading_module.value

    if object_registry and frame is not None:
        frame = frame.copy()
        for obj in object_registry.registry:
            x1, y1, x2, y2 = obj.xyxy.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID: {obj.guid} | Name: {obj.name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the annotated frame
    if frame is not None:
        cv2.imshow('Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        image_reading_module.stop()
        object_detection_module.stop()
        break

cv2.destroyAllWindows()
