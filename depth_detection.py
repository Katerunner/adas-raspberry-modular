import cv2
import numpy as np

from src.modules.depth_detection_module import DepthDetectionModule
from src.modules.image_reading_module import ImageReadingModule

CUTOUT = 72

video_path = "assets/videos/video_1.mp4"

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
depth_detection_module = DepthDetectionModule(source_module=image_reading_module)
frame_width = image_reading_module.frame_width
frame_height = image_reading_module.frame_height

dst_width = 256
dst_height = 1024

image_reading_module.start()
depth_detection_module.start()
while True:
    frame = image_reading_module.value

    if frame is None:
        continue

    depth_map = depth_detection_module.value

    if depth_map is not None:

        depth_mask = np.where(depth_map >= CUTOUT)
        high_y = np.max(depth_mask[0]) if depth_mask[0].size > 0 else int(frame_height * 0.47)

        src_points = np.float32([
            [0 - frame_width * 0.75, frame_height],
            [frame_width + frame_width * 0.75, frame_height],
            [frame_width * 0.35, high_y],
            [frame_width * 0.65, high_y]
        ])

        dst_points = np.float32([
            [0, dst_height],
            [dst_width, dst_height],
            [0, 0],
            [dst_width, 0]
        ])

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        bird_eye_view = cv2.warpPerspective(frame, M, (dst_width, dst_height))

        for point in src_points:
            cv2.circle(frame, tuple(point.astype(int)), 10, (0, 255, 0), -1)

        cv2.imshow("Perspective Transformation", bird_eye_view)

    cv2.imshow("Original Frame", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
