import numpy as np
from sklearn.linear_model import LinearRegression


class MovingObject:
    def __init__(self, guid: str, name: str, xyxy: np.ndarray, s: int, history_max: int = 20):
        self.guid = guid
        self.history_max = history_max
        self.name = name
        self.xyxy = xyxy
        self.s = s
        self.position_history = []

    def update_history(self, xyxy: np.ndarray, s: int):
        self.xyxy = xyxy
        self.s = s
        self.position_history.append((xyxy[0], xyxy[1], s))
        self.position_history = self.position_history[-self.history_max:]

    def predict_position(self, s_after: int):
        data = np.array(self.position_history)
        timeline = data[:, -1].reshape(-1, 1)  # Reshape timeline to 2D array
        position = data[:, :2]  # Extract x and y coordinates
        model = LinearRegression()
        model.fit(timeline, position)
        return model.predict([[timeline[-1, 0] + s_after]])  # Predict for reshaped input
