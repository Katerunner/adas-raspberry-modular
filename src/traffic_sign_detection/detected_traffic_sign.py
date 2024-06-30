from numpy import ndarray

from src.traffic_sign_detection.traffic_sign import TrafficSign


class DetectedTrafficSign:
    def __init__(self,
                 class_id,
                 confidence: float,
                 xyxyn: list[float],
                 traffic_sign: TrafficSign = None,
                 image: ndarray = None):
        self.class_id = class_id
        self.confidence = confidence
        self.xyxyn = xyxyn
        self.image = image
        self.traffic_sign = traffic_sign
        self.center_coords = self._get_center_coords()

    def _get_center_coords(self):
        return (self.xyxyn[0] + self.xyxyn[2]) / 2, (self.xyxyn[1] + self.xyxyn[3]) / 2
