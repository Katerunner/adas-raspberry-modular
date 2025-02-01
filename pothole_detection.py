import cv2
from ultralytics import YOLO

from src.modules.image_reading_module import ImageReadingModule
from src.modules.pothole_detection_module import PotholeDetectionModule

# Load the video
video_path = 'assets/videos/video_1.mp4'
object_model_path = 'trained_models/hole-yolov8n.pt'

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
object_detection_module = PotholeDetectionModule(
    source_module=image_reading_module,
    model_weights=object_model_path
)

# Start the modules
image_reading_module.start()
object_detection_module.start()

cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    pothole_registry = object_detection_module.value
    frame = image_reading_module.value

    if pothole_registry and frame is not None:
        frame = frame.copy()
        for obj in pothole_registry:
            x1, y1, x2, y2 = obj.xyxy.astype(int)
            cv2.line(frame, (x1, y2), (x2, y2), (0, 0, 255), 3)

            label = f"pothole {obj.conf:.2f}"
            cv2.putText(frame, label, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the annotated frame
    if frame is not None:
        cv2.imshow('Pothole Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        image_reading_module.stop()
        object_detection_module.stop()
        break

cv2.destroyAllWindows()
