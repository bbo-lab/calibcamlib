import numpy as np
from calibcamlib import distortion as dist  # TODO: make model variable
from bbo import vectorlib


class Camera:
    def __init__(self, A, k, xi=0, offset=None, distortion=None, projection_model='perspective'):  # TODO: Implement variable distortion
        if distortion is not None:
            raise ValueError("Distortion parameter is not implemented yet.")
        if offset is None:
            offset = [0, 0]

        self.offset = offset
        self.A = A.reshape(3, 3)
        self.k = k.reshape(5)
        self.xi = xi
        self.projection_model = projection_model

    def convert(self, xp, dtype=None):
        return Camera(
            vectorlib.convert(self.A, xp, dtype=dtype),
            vectorlib.convert(self.k, xp, dtype=dtype),
            xi=self.xi,
            offset=vectorlib.convert(self.offset, xp, dtype=dtype),
            projection_model=self.projection_model
        )

    def sensor_to_space(self, x:np.ndarray, offset=None) -> np.ndarray:
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

        if self.projection_model == 'fisheye_equidistant':
            xequidist, yequidist = X[:, (0,)], X[:, (1,)]
            radius2d = np.sqrt(np.square(xequidist) + np.square(yequidist))
            div = np.divide(np.sin(radius2d), radius2d, out=np.ones_like(radius2d), where=radius2d != 0)
            X[:, (0,)] = div * xequidist
            X[:, (1,)] = div * yequidist
            X[:, (2,)] = np.cos(radius2d)
        elif self.projection_model == "perspective" or self.projection_model is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                X /= np.linalg.norm(X, axis=-1, keepdims=True)
            if self.xi != 0:
                radicand = 1 + (X[:, (2,)] ** 2 - 1) * self.xi ** 2
                rad_mask = radicand.reshape((-1,)) >= 0

                a = np.full(X[:, (2,)].shape, np.nan)
                a[rad_mask, :] = self.xi * X[rad_mask, 2].reshape(-1, 1) + np.sqrt(radicand[rad_mask])

                X = X*a
                X[..., (2,)] = X[..., (2,)] - self.xi
        else:
            raise ValueError(f"Unknown projection model: {self.projection_model}")

        if len(x_shape) != 2:
            X = X.reshape(x_shape[:-1] + (3,))

        return X

    def _apply_projection_model(self, X: np.ndarray, xp, fastmath=False) -> np.ndarray:
        if self.projection_model == 'fisheye_equidistant':
            x = X[..., 0:2]
            xsq = xp.square(x)
            length = xp.sqrt(xsq[..., 0] + xsq[..., 1])
            elev = np.arctan2(length, X[..., 2])
            xp.divide(elev, length, where=length != 0, out=length)
            if fastmath:
                x = x * length[..., None]
            else:
                x = np.stack((x[..., 0] * length, x[..., 1] * length, xp.ones_like(X[..., 2])), axis=-1)
        elif self.projection_model is None or self.projection_model == 'perspective':
            if self.xi != 0:
                norm = xp.linalg.norm(X, axis=-1, keepdims=True)
                X = xp.where(norm == 0, X, X / norm)
                X[..., (2,)] += self.xi
            if fastmath:
                x = X[..., 0:2] / X[:, (2,)]
            else:
                x = X / X[:, (2,)]
        else:
            raise ValueError(f"Unknown projection model: {self.projection_model}")
        return x

    def space_to_sensor(self, X:np.ndarray, offset=None, check_inverse=False, fastmath=False) -> np.ndarray:
        if offset is None:
            offset = self.offset

        X_shape = X.shape
        if len(X_shape) != 2:
            X = X.reshape((-1, 3))

        original_space_coords = X

        xp = vectorlib.get_array_module(X)
        #if xp.all(xp.isnan(X)):
        #    return xp.full_like(X, shape=(*X_shape[:-1], 2), fill_value=xp.nan)

        x = self._apply_projection_model(X, xp, fastmath=fastmath)

        # Faster version same accuracy but not bitequal:
        if fastmath:
            x = dist.distort(x, self.k, fastmath=fastmath)
            x = x @ self.A.T[0:2, 0:2]
            x += self.A.T[2, 0:2] - xp.asarray(offset)
        else:
            x[:, 0:2] = dist.distort(x[:, 0:2], self.k, fastmath=fastmath)
            x = x @ self.A.T[:, 0:2]
            x = x - offset

        if check_inverse:
            space_loc = self.sensor_to_space(x, offset=offset)
            # set all locations that are not close to the mothds input to nan
            original_space_coords = original_space_coords / xp.linalg.norm(original_space_coords, axis=-1,
                                                                           keepdims=True)
            #space_coords = space_loc / np.linalg.norm(space_loc, axis=-1, keepdims=True)
            #nanmask = ~np.isnan(original_space_coords).any(axis=-1)
            close_mask = xp.linalg.norm(space_loc - original_space_coords, axis=-1) < 1e-5
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
            "projection_model": self.projection_model
        }