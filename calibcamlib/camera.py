import numpy as np
from calibcamlib import distortion as dist  # TODO: make model variable


class Camera:
    def __init__(self, A, k, xi=0, offset=None, distortion=None):  # TODO: Implement variable distortion
        if distortion is not None:
            raise ValueError("Distortion parameter is not implemented yet.")
        if offset is None:
            offset = [0, 0]

        self.offset = offset
        self.A = A.reshape(3, 3)
        self.k = k.reshape(5)
        self.xi = xi


    def sensor_to_space(self, x, offset=None):
        """
        Transforms 2D sensor coordinates into 3D space coordinates using intrinsic parameters
        and distortion correction.

        Parameters:
            x (np.ndarray): A 2D array of sensor coordinates with shape (..., 2).
            offset (np.ndarray, optional): Offset to adjust coordinates. Defaults to `self.offset`.

        Returns:
            np.ndarray: Transformed 3D space coordinates with shape (..., 3), or the corresponding
            reshaped structure matching the input if necessary.
        """
        if offset is None:
            offset = self.offset

        x_shape = x.shape
        if len(x_shape) != 2:
            x = x.reshape((-1, 2))

        if np.all(np.isnan(x)):
            return np.full(shape=x_shape[:-1] + (3,), fill_value=np.nan)
        # assert self.k[2] == 0 and self.k[3] == 0 and self.k[4] == 0
        x = x + offset

        X = np.empty(shape=(x.shape[0], 3))
        X[..., 0:2] = x
        X[..., 2] = 1

        X = X @ np.linalg.inv(self.A.T)

        X[:, 0:2] = dist.distort_inverse(X[:, 0:2], self.k)
        with np.errstate(divide='ignore', invalid='ignore'):
            X /= np.sqrt(np.sum(X ** 2, axis=-1, keepdims=True))
        radicand = 1 + (X[:, (2,)] ** 2 - 1) * self.xi ** 2
        rad_mask = radicand.reshape((-1,)) >= 0

        a = np.full(X[:, (2,)].shape, np.nan)
        a[rad_mask, :] = self.xi * X[rad_mask, 2].reshape(-1, 1) + np.sqrt(radicand[rad_mask])

        X = X*a
        X[..., (2,)] = X[..., (2,)] - self.xi

        if len(x_shape) != 2:
            X = X.reshape(x_shape[:-1] + (3,))

        return X

    def space_to_sensor(self, X, offset=None, check_inverse=False):
        if offset is None:
            offset = self.offset

        X_shape = X.shape
        if len(X_shape) != 2:
            X = X.reshape((-1, 3))

        original_space_coords = X

        if np.all(np.isnan(X)):
            return np.full((*X_shape[:-1], 2), np.nan)

        if not self.xi == 0:
            norm = np.linalg.norm(X, axis=-1, keepdims=True)
            X = np.where(norm == 0, X, X / norm)
            #X = np.divide(X, norm, where=norm!=0)
            X[..., (2,)] = X[..., (2,)] + self.xi


        # code from calibcam.multical_plot.project_board
        x = X / X[:, (2,)]
        x[:, 0:2] = dist.distort(x[:, 0:2], self.k)

        x = x @ self.A.T
        x = x[:, 0:2] - offset

        if check_inverse:
            space_loc = self.sensor_to_space(x, offset=offset)
            # set all locations that are not close to the mothds input to nan
            original_space_coords = original_space_coords / np.linalg.norm(original_space_coords, axis=-1,
                                                                           keepdims=True)
            space_coords = space_loc / np.linalg.norm(space_loc, axis=-1, keepdims=True)
            nanmask = ~np.isnan(original_space_coords).any(axis=-1)
            close_mask = np.linalg.norm(space_loc - original_space_coords, axis=-1) < 1e-5
            x[~close_mask] = np.nan

        if len(X_shape) != 2:
            x = x.reshape(X_shape[:-1]+(2,))

        return x

    def as_dict(self):
        return {
            "A": self.A,
            "k": self.k,
            "xi": self.xi,
            "offset": self.offset,
        }