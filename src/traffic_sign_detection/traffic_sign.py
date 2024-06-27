import numpy as np


class TrafficSign:
    def __init__(self, code: str, name: str, category: str, picture: np.ndarray):
        self.code = code
        self.name = name
        self.category = category
        self.picture = picture
