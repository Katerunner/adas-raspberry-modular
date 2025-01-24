import cv2
import numpy as np
from src.modules.depth_detection_module import DepthDetectionModule
from src.modules.image_reading_module import ImageReadingModule
from sklearn.linear_model import Ridge


def compute_frame_depth_data_with_filters(frame, depth_map, neighbor_distance=1):
    height, width, _ = frame.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    x_normalized = x_indices / width
    y_normalized = y_indices / height

    r = frame[:, :, 2] / 255.0
    g = frame[:, :, 1] / 255.0
    b = frame[:, :, 0] / 255.0

    kernel_size = 2 * neighbor_distance + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    r_neighbors = cv2.filter2D(r, -1, kernel)
    g_neighbors = cv2.filter2D(g, -1, kernel)
    b_neighbors = cv2.filter2D(b, -1, kernel)

    # Adjust the depth map and features to match the shape after convolution
    valid_h = height - 2 * neighbor_distance
    valid_w = width - 2 * neighbor_distance

    x_normalized = x_normalized[neighbor_distance:-neighbor_distance, neighbor_distance:-neighbor_distance].flatten()
    y_normalized = y_normalized[neighbor_distance:-neighbor_distance, neighbor_distance:-neighbor_distance].flatten()
    r_neighbors = r_neighbors[neighbor_distance:-neighbor_distance, neighbor_distance:-neighbor_distance].flatten()
    g_neighbors = g_neighbors[neighbor_distance:-neighbor_distance, neighbor_distance:-neighbor_distance].flatten()
    b_neighbors = b_neighbors[neighbor_distance:-neighbor_distance, neighbor_distance:-neighbor_distance].flatten()
    depth_flattened = depth_map[neighbor_distance:-neighbor_distance, neighbor_distance:-neighbor_distance].flatten()

    frame_depth_data = np.vstack((x_normalized, y_normalized, r_neighbors, g_neighbors, b_neighbors, depth_flattened)).T
    return frame_depth_data


def recreate_frame_from_prediction(frame, depth_prediction, frame_shape):
    depth_image = depth_prediction.reshape(frame_shape[:2])
    depth_normalized = np.uint8(np.clip(depth_image, 0, 100) / 100 * 255)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colormap


video_path = "assets/videos/video_1.mp4"

image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
depth_detection_module = DepthDetectionModule(source_module=image_reading_module)
image_reading_module.start()
depth_detection_module.start()

model = Ridge()
latest_prediction = None
frame_shape = None

neighbor_distance = 1

while True:
    frame = image_reading_module.value

    if frame is None:
        continue

    depth_map = depth_detection_module.value

    if depth_map is not None:
        frame_shape = frame.shape
        frame_depth_data = compute_frame_depth_data_with_filters(frame, depth_map, neighbor_distance)
        X = frame_depth_data[:, :-1]
        y = frame_depth_data[:, -1]
        model.fit(X, y)
        latest_prediction = model.predict(X)

    if latest_prediction is not None and frame_shape is not None:
        depth_colormap = recreate_frame_from_prediction(frame, latest_prediction, frame_shape)
        cv2.imshow("Recreated Depth Map", depth_colormap)

    cv2.imshow("Original Frame", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
