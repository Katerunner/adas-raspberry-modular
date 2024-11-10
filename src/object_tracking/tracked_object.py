import numpy as np


class TrackedObject:
    def __init__(
            self,
            tracking_id: int,
            features_array: np.ndarray,
            image: np.ndarray,
            position_array: np.ndarray,
            position_shift_array: np.ndarray
    ):
        self.tracking_id = tracking_id
        self.features_array = features_array
        self.image = image
        self.position_array = position_array
        self.position_shift_array = position_shift_array
