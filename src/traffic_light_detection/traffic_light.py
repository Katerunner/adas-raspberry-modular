import numpy as np


class TrafficLight:
    def __init__(self, xyxy: np.ndarray, color: str):
        self.xyxy = xyxy
        self.color = color

    def get_horizontal_position(self):
        return np.mean(self.xyxy[0] + self.xyxy[2])
