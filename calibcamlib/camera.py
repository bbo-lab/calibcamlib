import numpy as np
from calibcamlib import distortion as dist  # TODO: make model variable


class Camera:
    def __init__(self, A, k, xi=0, offset=None, distortion=None):  # TODO: Implement variable distortion
        if offset is None:
            offset = [0, 0]

        self.offset = offset
        self.A = A.reshape(3, 3)
        self.k = k.reshape(5)
        self.xi = xi

    def sensor_to_space(self, x, offset=None):
        if offset is None:
            offset = self.offset

        # assert self.k[2] == 0 and self.k[3] == 0 and self.k[4] == 0
        x = x + offset

        X = np.zeros(shape=(x.shape[0], 3))
        X[:, 0:2] = x
        X[:, 2] = 1

        X = X @ np.linalg.inv(self.A.T)

        X[:, 0:2] = dist.distort_inverse(X[:, 0:2], self.k)

        X /= np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
        a = self.xi * X[:, (2,)] + np.sqrt(1 + (X[:, (2,)] ** 2 - 1) * self.xi ** 2)
        X = X*a
        X[..., (2,)] = X[..., (2,)] - self.xi

        return X

    def space_to_sensor(self, X, offset=None):
        if offset is None:
            offset = self.offset

        if not self.xi == 0:
            norm = np.linalg.norm(X, axis=-1, keepdims=True)
            X = np.where(norm == 0, X, X / norm)
            X[..., (2,)] = X[..., (2,)] + self.xi

        # code from calibcam.multical_plot.project_board
        x = X / X[:, 2, np.newaxis]

        x[:, 0:2] = dist.distort(x[:, 0:2], self.k)

        x = x @ self.A.T

        return x[:, 0:2] - offset

