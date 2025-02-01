import numpy as np


class Pothole:
    def __init__(self, xyxy: np.ndarray, conf: float):
        self.xyxy = xyxy
        self.conf = conf
