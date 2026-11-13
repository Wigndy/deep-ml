import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)

    X_trans = X_arr.T
    theta = np.linalg.inv(X_trans @ X_arr) @ X_trans @ y_arr
    theta = np.round(theta, 4).tolist()
    return theta