import time
import uuid

from src.traffic_light_detection.traffic_light import TrafficLight

TRAFFIC_LIGHT_CLASSES = ["traffic_light_r", "traffic_light_y", "traffic_light_g"]


class TrafficLightRegistry:
    def __init__(self, max_lifetime: int = 1):
        self.max_lifetime = max_lifetime
        self._registry_objs: list[TrafficLight] = []
        self._registry_time: list[bool] = []

    @property
    def registry(self):
        return self._registry_objs

    def clean_old_lights(self):
        current_time = time.time()
        filtered_data = [(obj, reg_time) for obj, reg_time in zip(self._registry_objs, self._registry_time)
                         if current_time - reg_time <= self.max_lifetime]

        self._registry_objs, self._registry_time = zip(*filtered_data) if filtered_data else ([], [])

    def update_registry(self, traffic_light: TrafficLight):
        self.clean_old_lights()
        self._registry_objs.append(traffic_light)
        self._registry_time.append(time.time())

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
                if names[i] not in TRAFFIC_LIGHT_CLASSES:
                    continue

                traffic_light = TrafficLight(
                    xyxy=xyxys[i],
                    color=names[i][-1]
                )

                self.update_registry(traffic_light=traffic_light)
