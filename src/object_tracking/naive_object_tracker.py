import time

import cv2
import numpy as np

from src.object_tracking.tracked_object import TrackedObject


class NaiveObjectTracker:
    def __init__(self,
                 max_objects=None,
                 confidence_threshold=0.8,
                 feature_threshold=0.95,
                 position_threshold=0.95,
                 lifespan=3):

        self.max_objects = max_objects
        self.feature_threshold = feature_threshold
        self.position_threshold = position_threshold
        self.confidence_threshold = confidence_threshold
        self.lifespan = lifespan

        self.memory = []
        self.lifespan_array = []
        self._incremental_id = 0

    @property
    def next_id(self):
        id_to_return = self._incremental_id
        self._incremental_id += 1
        return id_to_return

    def remove_expired_objects(self):
        current_time = time.time()
        for i in reversed(range(len(self.memory))):
            lifespan_value = self.lifespan_array[i]
            if current_time - lifespan_value > self.lifespan:
                del self.memory[i]
                del self.lifespan_array[i]

    def _get_objects_from_yolo_results(self, prediction_result):
        tracked_objects = []

        original_image = prediction_result.orig_img
        boxes = prediction_result.boxes
        if boxes is not None:
            confs = boxes.conf.numpy()
            xyxys = boxes.xyxy.numpy()

            for i in range(len(confs)):
                features_array = self._calculate_features(original_image, xyxys[i])
                position_array = self._calculate_position(xyxys[i], original_image.shape[1], original_image.shape[0])
                position_shift_array = np.zeros_like(position_array)

                x1, y1, x2, y2 = xyxys[i].astype(int)
                cropped_image = original_image[y1:y2, x1:x2]

                tracked_objects.append(
                    TrackedObject(
                        tracking_id=None,
                        features_array=features_array,
                        image=cropped_image,
                        position_array=position_array,
                        position_shift_array=position_shift_array
                    )
                )
        return tracked_objects

    def process_yolo_result(self, prediction_result):
        tracked_objects = self._get_objects_from_yolo_results(prediction_result)
        object_ids = [self.add_or_update_object(tracked_object).tracking_id for tracked_object in tracked_objects]
        return object_ids

    def compare_object(self, tracked_object: TrackedObject):
        best_match = -1
        best_simil = 0
        for i in range(len(self.memory)):
            object_to_compare = self.memory[i]
            features_sim = 1 - self._mape(tracked_object.features_array, object_to_compare.features_array)
            positions_sim = 1 - self._mae(tracked_object.position_array, object_to_compare.position_array)
            if (features_sim > self.feature_threshold) and (positions_sim > self.position_threshold):
                simil = features_sim + positions_sim
                if simil > best_simil:
                    best_match = i
                    best_simil = features_sim + positions_sim
        return best_match

    def add_object(self, tracked_object: TrackedObject):

        if self.max_objects is not None and len(self.memory) == self.max_objects:
            self.memory.pop(0)
            self.lifespan_array.pop(0)

        tracked_object.tracking_id = self.next_id
        self.memory.append(tracked_object)
        self.lifespan_array.append(time.time())
        return tracked_object

    def update_object(self, memory_index: int, tracked_object: TrackedObject):
        position_shift_array = np.abs(self.memory[memory_index].position_shift_array - tracked_object.position_array)
        tracked_object.tracking_id = self.memory[memory_index].tracking_id  # Keep previous id
        tracked_object.position_shift_array = position_shift_array
        self.memory[memory_index] = tracked_object
        self.lifespan_array[memory_index] = time.time()  # Update life
        return tracked_object

    def add_or_update_object(self, tracked_object: TrackedObject):
        self.remove_expired_objects()

        memory_index = self.compare_object(tracked_object=tracked_object)

        if memory_index == -1:
            return self.add_object(tracked_object=tracked_object)

        return self.update_object(memory_index=memory_index, tracked_object=tracked_object)

    @staticmethod
    def _calculate_features(image, xyxy):
        x1, y1, x2, y2 = xyxy.astype(int)
        cropped_image = image[y1:y2, x1:x2]

        # Mean RGB
        mean_rgb = np.mean(cropped_image, axis=(0, 1))

        # Mean of all RGB channels
        mean_all = np.mean(mean_rgb)

        # Standard deviation of RGB channels
        std_rgb = np.std(cropped_image, axis=(0, 1))

        # Dimensions
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Aspect ratio
        aspect_ratio = width / height if height != 0 else 0

        # Color histogram (normalized, 8 bins per channel)
        hist_r = cv2.calcHist([cropped_image], [0], None, [8], [0, 256]).flatten()
        hist_g = cv2.calcHist([cropped_image], [1], None, [8], [0, 256]).flatten()
        hist_b = cv2.calcHist([cropped_image], [2], None, [8], [0, 256]).flatten()
        hist = np.concatenate([hist_r, hist_g, hist_b])
        hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist

        # Sobel gradient magnitude (mean)
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        mean_gradient = np.mean(gradient_magnitude)

        # Compactness (perimeter^2 / area)
        perimeter = 2 * (width + height)
        compactness = (perimeter ** 2) / area if area > 0 else 0

        # Combine features
        features = np.array([
            *mean_rgb,  # 3 features
            mean_all,  # 1 feature
            *std_rgb,  # 3 features
            width / 2,  # 1 feature
            height / 2,  # 1 feature
            area / 2,  # 1 feature
            aspect_ratio / 2,  # 1 feature
            mean_gradient,  # 1 feature
            compactness / 2,  # 1 feature
            *hist_normalized  # 24 features (8 bins * 3 channels)
        ])
        return features

    @staticmethod
    def _calculate_position(xyxy, image_width, image_height):
        x1, y1, x2, y2 = xyxy
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        normalized_center_x = center_x / image_width
        normalized_center_y = center_y / image_height
        return np.array([normalized_center_x, normalized_center_y])

    @staticmethod
    def _mape(y_true, y_pred, eps=100):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        error = np.abs((y_true - y_pred) / (y_true + eps))
        return np.mean(error)

    @staticmethod
    def _mae(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
