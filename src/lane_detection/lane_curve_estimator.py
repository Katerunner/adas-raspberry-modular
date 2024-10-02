import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class LaneCurveEstimator:
    def __init__(self, poly_degree=2, ransac_min_samples=2, ransac_loss="squared_error", n_points=10):
        self.poly_degree = poly_degree
        self.ransac_min_samples = ransac_min_samples
        self.ransac_loss = ransac_loss
        self.n_points = n_points

    def predict_lane_points(self, points, weights=None):
        if len(points) < 3:
            return []

        x_orig = np.array([p[0] for p in points])
        y_orig = np.array([p[1] for p in points])

        y_min = np.min(y_orig)
        y_max = np.max(y_orig)

        model = make_pipeline(
            PolynomialFeatures(self.poly_degree),
            RANSACRegressor(min_samples=self.ransac_min_samples, loss=self.ransac_loss)
        )

        # Fit the model with optional weights (confidences)
        model.fit(y_orig.reshape(-1, 1), x_orig, ransacregressor__sample_weight=weights)

        y_pred = np.linspace(y_min, y_max, self.n_points)
        X_pred = model.predict(y_pred.reshape(-1, 1))

        return np.array([(int(x), int(y)) for x, y in zip(X_pred, y_pred)])

    def __repr__(self):
        return (f"LaneCurveEstimator(poly_degree={self.poly_degree}, "
                f"ransac_min_samples={self.ransac_min_samples}, "
                f"ransac_loss={self.ransac_loss}, n_points={self.n_points})")
