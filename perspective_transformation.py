import cv2
import numpy as np

from src.modules.perspective_transformation_module import PerspectiveTransformationModule
from src.modules.image_reading_module import ImageReadingModule

video_path = "assets/videos/video_1.mp4"

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
perspective_transformation_module = PerspectiveTransformationModule(source_module=image_reading_module)

# Start the modules
image_reading_module.start()
perspective_transformation_module.start()

while True:

    if image_reading_module.value is not None:
        input_image = image_reading_module.value

        x, y = np.random.normal(0, 0.1, 2) / 10 + 0.5
        y *= input_image.shape[0]
        x *= input_image.shape[1]
        x, y = int(x), int(y)

        point_color = (0, 255, 0)  # Green color for the point
        point_radius = 5
        cv2.circle(input_image, (x, y), point_radius, point_color, -1)
        cv2.imshow('Original Frame', input_image)

    if perspective_transformation_module.value is not None:
        output_image = perspective_transformation_module.value
        x_t, y_t = perspective_transformation_module.transform_point(x=x, y=y)
        x_t, y_t = int(x_t), int(y_t)

        point_color = (255, 0, 255)  # Green color for the point
        point_radius = 10
        cv2.circle(output_image, (x_t, y_t), point_radius, point_color, -1)
        cv2.imshow('Bird\'s Eye View', output_image)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

image_reading_module.stop()
perspective_transformation_module.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()
