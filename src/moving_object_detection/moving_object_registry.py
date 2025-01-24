import time
import uuid

from src.moving_object_detection.moving_object import MovingObject

MOVING_OBJECT_CLASSES = ["car", "truck", "pedestrian", "bus", "train", "motorcycle", "bicycle"]


class MovingObjectRegistry:
    def __init__(self, max_objects: int = 20):
        self.max_objects = max_objects
        self._registry_objs: list[MovingObject] = []
        self._registry_ids: list[str] = []
        self._registry_fresh: list[bool] = []

    @property
    def registry(self):
        return [obj for obj, fresh in zip(self._registry_objs, self._registry_fresh) if fresh]

    def update_registry(self, moving_object: MovingObject):
        if moving_object.guid in self._registry_ids:
            object_index = self._registry_ids.index(moving_object.guid)
            self._registry_objs[object_index].update_history(xyxy=moving_object.xyxy, s=moving_object.s)
            self._registry_fresh[object_index] = True
        else:
            self._registry_ids.append(moving_object.guid)
            self._registry_objs.append(moving_object)
            self._registry_fresh.append(True)

            self._registry_ids = self._registry_ids[-self.max_objects:]
            self._registry_objs = self._registry_objs[-self.max_objects:]
            self._registry_fresh = self._registry_fresh[-self.max_objects:]

    def from_yolo_result(self, prediction_result, ids: list = None):
        self._registry_fresh = [False] * len(self._registry_fresh)

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
