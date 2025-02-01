import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge


class MovingObject:
    def __init__(self, guid: str, name: str, xyxy: np.ndarray, s: int, history_max: int = 10):
        self.guid = guid
        self.history_max = history_max
        self.name = name
        self.xyxy = xyxy
        self.s = s
        self.position_history = []
        self.weights = np.arange(history_max) / history_max + 1

    def update_history(self, xyxy: np.ndarray, s: int):
        self.xyxy = xyxy
        self.s = s
        x_center = np.mean([xyxy[0], xyxy[2]])
        y_bottom = xyxy[3]
        self.position_history.append((x_center, y_bottom, s))
        self.position_history = self.position_history[-self.history_max:]

    def predict_position(self, s_after: int, exact_time: int = None):
        x_center = np.mean([self.xyxy[0], self.xyxy[2]])
        y_bottom = self.xyxy[3]

        if len(self.position_history) < 2:
            return x_center, y_bottom

        try:
            data = np.array(self.position_history)
            timeline = data[:, -1].reshape(-1, 1)  # Reshape timeline to 2D array
            position = data[:, :2]  # Extract x and y coordinates
            weights = self.weights[-len(data):]
            model = RANSACRegressor(estimator=LinearRegression(), stop_probability=0.99, max_trials=50)
            model.fit(X=timeline, y=position, sample_weight=weights)

            prediction_time = timeline[-1, 0] + s_after if exact_time is None else exact_time
            return model.predict([[prediction_time]])[0]  # Predict for reshaped input
        except Exception as e:
            return x_center, y_bottom
