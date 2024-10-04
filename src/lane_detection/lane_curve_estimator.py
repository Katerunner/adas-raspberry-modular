import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings


class LaneCurveEstimator:
    def __init__(
            self,
            image_shape,
            poly_degree=2,
            ransac_min_samples=2,
            ransac_loss="squared_error",
            n_points=10,
            memory_size=100,
            use_weights=True,
            decay=2
    ):
        self.image_shape = image_shape  # (H, W)
        self.poly_degree = poly_degree
        self.ransac_min_samples = ransac_min_samples
        self.ransac_loss = ransac_loss
        self.n_points = n_points
        self.memory_size = memory_size
        self.use_weights = use_weights
        self.decay = decay
        self.lane_memory = {
            'LL': {'points': [], 'weights': []},
            'LC': {'points': [], 'weights': []},
            'RC': {'points': [], 'weights': []},
            'RR': {'points': [], 'weights': []}
        }

    def predict_lane_points(self, lane_type, new_points, new_weights=None):
        self._validate_lane_type(lane_type)
        new_weights = self._handle_weights(lane_type, new_points, new_weights)
        self._decay_weights(lane_type)
        self._update_memory(lane_type, new_points, new_weights)
        y_min, y_max = self._get_global_y_min_max()

        return self._perform_prediction(lane_type, y_min, y_max)

    def _validate_lane_type(self, lane_type):
        if lane_type not in self.lane_memory:
            raise ValueError(f"Invalid lane type: {lane_type}")

    def _handle_weights(self, lane_type, new_points, new_weights):
        if new_weights is None:
            if self.use_weights:
                warnings.warn(f"Weights not provided for {lane_type}, defaulting to 1 for all new points.")
                return [1] * len(new_points)
            else:
                return []
        return new_weights

    def _decay_weights(self, lane_type):
        self.lane_memory[lane_type]['weights'] = [w / self.decay for w in self.lane_memory[lane_type]['weights']]

    def _update_memory(self, lane_type, new_points, new_weights):
        self.lane_memory[lane_type]['points'].extend(new_points)
        if self.use_weights:
            self.lane_memory[lane_type]['weights'].extend(new_weights)

        if len(self.lane_memory[lane_type]['points']) > self.memory_size:
            excess = len(self.lane_memory[lane_type]['points']) - self.memory_size
            self.lane_memory[lane_type]['points'] = self.lane_memory[lane_type]['points'][-self.memory_size:]
            if self.use_weights:
                self.lane_memory[lane_type]['weights'] = self.lane_memory[lane_type]['weights'][-self.memory_size:]

    def _get_global_y_min_max(self):
        y_values = []
        for lane in self.lane_memory.values():
            y_values.extend([p[1] for p in lane['points']])

        if not y_values:
            return 0, 0  # Return default if no points exist in any lane

        return np.min(y_values), np.max(y_values)

    def _perform_prediction(self, lane_type, y_min, y_max):
        points = self.lane_memory[lane_type]['points']
        weights = self.lane_memory[lane_type]['weights'] if self.use_weights else None

        if len(points) < 3:
            return []

        x_orig = np.array([p[0] for p in points])
        y_orig = np.array([p[1] for p in points])

        model = make_pipeline(
            PolynomialFeatures(self.poly_degree),
            RANSACRegressor(min_samples=self.ransac_min_samples, loss=self.ransac_loss)
        )

        if self.use_weights and weights is not None:
            model.fit(y_orig.reshape(-1, 1), x_orig, ransacregressor__sample_weight=weights)
        else:
            model.fit(y_orig.reshape(-1, 1), x_orig)

        y_pred = np.linspace(y_min, y_max, self.n_points)
        X_pred = model.predict(y_pred.reshape(-1, 1))

        return np.array([(int(x), int(y)) for x, y in zip(X_pred, y_pred)])

    def __repr__(self):
        return (f"LaneCurveEstimator(image_shape={self.image_shape}, poly_degree={self.poly_degree}, "
                f"ransac_min_samples={self.ransac_min_samples}, ransac_loss={self.ransac_loss}, "
                f"n_points={self.n_points}, memory_size={self.memory_size}, use_weights={self.use_weights}, "
                f"decay={self.decay})")
