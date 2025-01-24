import cv2
import numpy as np
from src.modules.base_module import BaseModule

DEFAULT_DST_WIDTH = 256
DEFAULT_DST_HEIGHT = 1024

DEFAULT_SRC_WEIGHTS = np.float32([
    [0.0, 1.0],  # Bottom-left
    [1.0, 1.0],  # Bottom-right
    [0.47, 0.47],  # Top-left (near center)
    [0.53, 0.47]  # Top-right (near center)
])

DEFAULT_DST_WEIGHTS = np.float32([
    [0.0, 1.0],  # Bottom-left
    [1.0, 1.0],  # Bottom-right
    [0.0, 0.0],  # Top-left (near center)
    [1.0, 0.0]  # Top-right (near center)
])


class PerspectiveTransformationModule(BaseModule):
    def __init__(
            self, source_module: BaseModule,
            src_weights: np.ndarray = DEFAULT_SRC_WEIGHTS,
            dst_weights: np.ndarray = DEFAULT_DST_WEIGHTS,
            dst_width: int = DEFAULT_DST_WIDTH,
            dst_height: int = DEFAULT_DST_HEIGHT
    ):
        super().__init__()

        self.source_module = source_module

        self.dst_height = dst_height
        self.dst_width = dst_width

        self.dst_weights = dst_weights
        self.src_weights = src_weights

        self.M = self.compute_transform_matrix()

    def compute_transform_matrix(self):
        if self.source_module.value is None:
            return None

        frame_width, frame_height = self.source_module.value.shape[:2]

        src_points = self.src_weights
        src_points[:, 0] *= frame_height
        src_points[:, 1] *= frame_width

        dst_points = self.dst_weights
        dst_points[:, 0] *= self.dst_width
        dst_points[:, 1] *= self.dst_height

        return cv2.getPerspectiveTransform(src_points, dst_points)

    def transform_point(self, x, y):
        src_point = np.array([x, y, 1], dtype=np.float32)
        transformed_point = np.dot(self.M, src_point)
        transformed_point /= transformed_point[2]  # Normalize by w
        return transformed_point[0], transformed_point[1]  # x', y'

    def perform(self):
        while not self._stop_event.is_set():
            frame = self.source_module.value

            if frame is None:
                continue

            if self.M is None:
                self.M = self.compute_transform_matrix()

            bird_eye_view = cv2.warpPerspective(frame, self.M, (self.dst_width, self.dst_height))
            self.value = bird_eye_view
