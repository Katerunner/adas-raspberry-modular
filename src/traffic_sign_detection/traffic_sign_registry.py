import time
import uuid

import numpy as np
from ultralytics import YOLO

from src.traffic_sign_detection.traffic_sign import TrafficSign


class TrafficSignRegistry:
    def __init__(self,
                 yolo_class_weights: str,
                 max_size: int = 8,
                 min_occurrences: int = 5,
                 casting_lifetime: int = 3,
                 registry_lifetime: int = 5):
        self.classificator = YOLO(yolo_class_weights, task='classify')
        self.max_size = max_size

        self._registry = []
        self._registry_time = []
        self._registry_guid = []

        self._casting = dict()

        self.registry_lifetime = registry_lifetime
        self.casting_lifetime = casting_lifetime
        self.min_occurrences = min_occurrences

    def _refresh_casting(self):
        guids_to_delete = [
            guid for guid in self._casting
            if self._casting[guid]["time"] < (time.time() - self.casting_lifetime)
        ]
        for guid in guids_to_delete:
            del self._casting[guid]

    def _refresh_registry(self):
        indices_to_delete = [
            i for i in range(len(self._registry))
            if self._registry_time[i] < (time.time() - self.registry_lifetime)
        ]
        for i in sorted(indices_to_delete, reverse=True):
            del self._registry[i]
            del self._registry_guid[i]
            del self._registry_time[i]

    def classify_traffic_sign(self, image: np.ndarray) -> str:
        results = self.classificator.predict(image, verbose=False)
        if not results:
            return "Unknown"

        pred_idx = results[0].probs.top1
        return self.classificator.names[pred_idx] if hasattr(self.classificator, "names") else str(pred_idx)

    @property
    def registry(self):
        self._refresh_registry()
        return self._registry

    def record_occurrence(self, traffic_sign: TrafficSign):
        self._refresh_casting()

        if traffic_sign.guid in self._registry_guid:
            self._registry[self._registry_guid.index(traffic_sign.guid)] = traffic_sign
            return

        if traffic_sign.guid not in self._casting:
            self._casting[traffic_sign.guid] = {"time": time.time(), "occurrences": 0}
            return

        self._casting[traffic_sign.guid]["occurrences"] += 1

        if self._casting[traffic_sign.guid]["occurrences"] >= self.min_occurrences:
            self._registry.append(traffic_sign)
            self._registry_guid.append(traffic_sign.guid)
            self._registry_time.append(time.time())

    def from_yolo_results(self, prediction_result, ids: list = None):
        original_image = prediction_result.orig_img
        boxes = prediction_result.boxes

        if boxes is not None:
            confs = boxes.conf.numpy()
            xyxys = boxes.xyxy.numpy()
            num_signs = len(confs)

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(num_signs)]

            assert len(ids) == num_signs

            for i in range(num_signs):
                x1, y1, x2, y2 = xyxys[i].astype(int)
                cropped_image = original_image[y1:y2, x1:x2]
                name = self.classify_traffic_sign(image=cropped_image)
                traffic_sign = TrafficSign(
                    name=name,
                    guid=ids[i],
                    image=cropped_image,
                    position=xyxys[i].astype(int)
                )
                self.record_occurrence(traffic_sign=traffic_sign)
