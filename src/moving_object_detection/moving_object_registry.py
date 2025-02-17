import threading
import time
import uuid

from src.moving_object_detection.moving_object import MovingObject

MOVING_OBJECT_CLASSES = ["car", "truck", "pedestrian", "bus", "train", "motorcycle", "bicycle"]


class MovingObjectRegistry:
    def __init__(self, max_lifetime: int = 1):
        self.max_lifetime = max_lifetime
        self._registry_objs: list[MovingObject] = []
        self._registry_ids: list[str] = []
        self._registry_times: list[float] = []

    @property
    def registry(self):
        self.clean_dead()
        return self._registry_objs

    def clean_dead(self):
        alive = [(time.time() - birth) < self.max_lifetime for birth in self._registry_times]

        self._registry_ids = [i for i, a in zip(self._registry_ids, alive) if a]
        self._registry_objs = [i for i, a in zip(self._registry_objs, alive) if a]
        self._registry_times = [i for i, a in zip(self._registry_times, alive) if a]

    def update_registry(self, moving_object: MovingObject):
        with threading.Lock():
            if moving_object.guid in self._registry_ids:
                object_index = self._registry_ids.index(moving_object.guid)
                self._registry_objs[object_index].update_history(xyxy=moving_object.xyxy, s=moving_object.s)
                self._registry_times[object_index] = time.time()
            else:
                self._registry_ids.append(moving_object.guid)
                self._registry_objs.append(moving_object)
                self._registry_times.append(time.time())

    def from_yolo_result(self, prediction_result, ids: list = None):
        name_dict = prediction_result.names
        boxes = prediction_result.boxes

        if boxes is not None:
            names = [name_dict.get(int(cls), "other") for cls in boxes.cls.numpy()]
            confs = boxes.conf.numpy()
            xyxys = boxes.xyxy.numpy()
            num_objects = len(confs)

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(num_objects)]

            assert len(ids) == num_objects

            for i in range(num_objects):
                if names[i] not in MOVING_OBJECT_CLASSES:
                    continue

                moving_object = MovingObject(
                    name=names[i],
                    guid=ids[i],
                    xyxy=xyxys[i],
                    s=time.time()
                )

                self.update_registry(moving_object=moving_object)

        self.clean_dead()
