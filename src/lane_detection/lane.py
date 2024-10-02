import numpy as np


class Lane:
    def __init__(self, points=None, confs=None, estimated_points=None):
        if points is None:
            points = []
        if confs is None:
            confs = []
        if estimated_points is None:
            estimated_points = []

        if len(points) != len(confs):
            raise ValueError("Length of points and confidences must be the same")

        self.points = np.array(points)
        self.confs = np.array(confs)
        self.estimated_points = np.array(estimated_points)

    def __repr__(self):
        return (f"Lane(points={self.points}, confs={self.confs}, "
                f"estimated_points={self.estimated_points})")
