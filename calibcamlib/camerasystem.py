import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa
from . import Camera
from .helper import intersect


# R,t are world->cam

class Camerasystem:
    def __init__(self):
        self.cameras = list()

    def add_camera(self, A, k, R, t):
        self.cameras.append({'camera': Camera(A, k), 'R': R, 't': t})

    def project(self, X):
        X_shape = X.shape
        X = X.reshape(-1, 3)
        x = np.zeros(shape=(len(self.cameras), X.shape[0], 2))

        for i, c in enumerate(self.cameras):
            coords_cam = (c['R'] @ X.T).T + c['t']
            x[i] = c['camera'].space_to_sensor(coords_cam).T.T

        return x.reshape((len(self.cameras),)+X_shape[0:-1]+(2,))

    def triangulate_3derr(self, x):
        # TODO support more than one point!
        V = np.empty(shape=(x.shape[0], x.shape[1], 3))
        P = np.empty(shape=(x.shape[0], x.shape[1], 3))

        for i, c in enumerate(self.cameras):
            V[i, :] = c['camera'].sensor_to_space(x[i]) @ c['R']
            P[i, :] = np.tile(-c['R'].T @ c['t'], (x.shape[1], 1))

        X = np.empty(V.shape[1:])
        X[:] = np.NaN
        for i, Xp in enumerate(X):
            if np.sum(~np.isnan(V[:, i, 1])) > 1:
                X[i] = intersect(P[:, i, :], V[:, i, :]).T

        return X, P, V

    @staticmethod
    def from_calibcam_file(filename: str):
        cs = Camerasystem()
        calib = np.load(filename, allow_pickle=True).item()

        for i in range(len(calib['RX1_fit'])):
            A = np.array([
                [calib['A_fit'][i][0], 0, calib['A_fit'][i][1]],
                [0, calib['A_fit'][i][2], calib['A_fit'][i][3]],
                [0, 0, 1]
            ])

            cs.add_camera(A,
                          calib['k_fit'][i],
                          calib['RX1_fit'][i],
                          calib['tX1_fit'][i] * calib['square_size_real']
                          )

        return cs

    @staticmethod
    def from_calibs(calibs):
        cs = Camerasystem()

        for calib in calibs:
            cs.add_camera(calib['A'],
                          calib['k'],
                          R.from_rotvec(calib['rvec_cam'].reshape((3,))).as_matrix(),
                          calib['tvec_cam'].reshape(1, 3)
                          )

        return cs
