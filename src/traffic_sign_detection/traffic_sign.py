import numpy as np


class TrafficSign:
    def __init__(self, guid: str, name: str, position: np.ndarray, image: np.ndarray):
        self.name = name
        self.guid = guid
        self.image = image
        self.position = position
