import numpy as np
from scipy.spatial.transform import Rotation as R
from calibcamlib import Camera
from calibcamlib.helper import intersect, get_line_dist

# R,t are world->cam
class Camerasystem:
    def __init__(self):
        self.cameras = list()

    def add_camera(self, A, k, rotmat, t):
        self.cameras.append({'camera': Camera(A, k), 'R': rotmat, 't': t})

    def project(self, X, offsets=None):
        if offsets is None:
            offsets = np.zeros((len(self.cameras), 2))

        X_shape = X.shape
        X = X.reshape(-1, 3)
        x = np.zeros(shape=(len(self.cameras), X.shape[0], 2))

        for i, (c, o) in enumerate(zip(self.cameras, offsets)):
            coords_cam = (c['R'] @ X.T).T + c['t']
            x[i] = c['camera'].space_to_sensor(coords_cam, o).T.T

        return x.reshape((len(self.cameras),) + X_shape[0:-1] + (2,))

    def get_camera_lines(self, x, offsets):
        if offsets is None:
            offsets = np.zeros((len(self.cameras), 2))

        x_shape = x.shape
        x = x.reshape((x_shape[0], -1, 2))

        V = np.empty(shape=(x.shape[0], x.shape[1], 3))
        P = np.empty(shape=(x.shape[0], x.shape[1], 3))

        for i, (c, o) in enumerate(zip(self.cameras, offsets)):
            V[i, :], P[i, :] = self.get_camera_lines_cam(x[i], i, o)

        return P.reshape(x_shape[0:-1] + (3,)), V.reshape(x_shape[0:-1] + (3,))

    def get_camera_lines_cam(self, x, cam_idx, offset):
        if offset is None:
            offset = np.zeros((1, 2))

        c = self.cameras[cam_idx]
        V = c['camera'].sensor_to_space(x, offset) @ c['R']
        P = np.tile(- c['t'] @ c['R'], (x.shape[0], 1))

        return V, P

    def triangulate_3derr(self, x, offsets):
        if offsets is None:
            offsets = np.zeros((len(self.cameras), 2))

        x_shape = x.shape
        x = x.reshape((x_shape[0], -1, 2))

        V, P = self.get_camera_lines(x, offsets)

        X = np.empty(V.shape[1:])
        X[:] = np.NaN
        for i, Xp in enumerate(X):
            if np.sum(~np.isnan(V[:, i, 1])) > 1:
                X[i] = intersect(P[:, i, :], V[:, i, :]).T

        return X.reshape(x_shape[1:-1] + (3,))

    def triangulate(self, x, offsets):
        # This will be changed to reprojection error in the future!
        # Use this function only if that is desired, else use triangulate_3derr
        return self.triangulate_3derr(x, offsets)

    def triangulate_nopointcorr(self, AB, offsets, linedist_thres, max_points=12):
        # Triangulates a 3d point from list of unsorted 2d points, automatically finding correspondences
        n_AB = np.array([ab.shape[0] for ab in AB])
        if np.sum(n_AB > 0) < 2:
            return np.zeros((0, 3))

        main_cam_idx = np.where(n_AB == np.max(n_AB[n_AB <= max_points]))[0][0]

        cam_bases = np.empty((len(AB), 3))
        cam_bases[:] = np.NaN

        full_ab = np.empty((len(AB), n_AB[main_cam_idx], 2))
        full_ab[:] = np.NaN
        full_ab[main_cam_idx] = AB[main_cam_idx]

        full_dirs = np.empty((len(AB), n_AB[main_cam_idx], 3))
        full_dirs[:] = np.NaN
        full_dirs[main_cam_idx, :, :], cb = self.get_camera_lines_cam(AB[main_cam_idx],
                                                                      main_cam_idx,
                                                                      offsets[main_cam_idx])
        cam_bases[main_cam_idx, :] = cb[0]

        for (iC, (ab, offset)) in enumerate(zip(AB, offsets)):
            if iC == main_cam_idx:
                continue
            if ab.shape[0] == 0:
                continue
            dirs, cb = self.get_camera_lines_cam(ab, iC, offset)

            cam_bases[iC] = cb[0]

            distances = np.array([[get_line_dist(cam_bases[main_cam_idx], full_dirs[main_cam_idx, i, :], cam_bases[iC],
                                                 dirs[j]) for j in range(dirs.shape[0])] for i in
                                  range(full_dirs.shape[1])])

            np.equal(distances, np.min(distances, axis=0)[np.newaxis, :])
            connectmat = np.all([distances < linedist_thres,
                                 np.equal(distances, np.min(distances, axis=1)[:, np.newaxis]),
                                 # np.equal(distances,np.min(distances,axis=0)[np.newaxis,:])
                                 ], axis=0)
            connectmat[:, np.sum(connectmat, axis=0) > 1] = False  # Discard ambiguities
            corrs = np.array(np.where(connectmat))

            if corrs.shape[1] == 0:
                continue

            full_ab[iC, corrs[0], :] = ab[corrs[1]]
            # print(full_ab[iC])
            full_dirs[iC, corrs[0], :] = dirs[corrs[1].T]
        if full_dirs.shape[1] == 0:
            return np.zeros((0, 3))

        points = np.array([intersect(cam_bases, full_dirs[:, i, :]) for i in range(full_dirs.shape[1])])
        points = points[~np.any(np.isnan(points), axis=1)]

        print(f"Triangulated {points.shape[0]} points")

        return points

    @staticmethod
    def from_calibcam_file(filename: str):
        cs = Camerasystem()
        calib = np.load(filename, allow_pickle=True)[()]

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

    @staticmethod
    def from_mcl(mc):
        cs = Camerasystem()

        for i_cal, cal in enumerate(mc['cal']):
            sc = cal['scaling']
            sp = cal['scale_pixels']
            ic = cal['icent']
            ss = cal['sensorsize']
            cam_matrix = np.array([
                [sc[0]*sp, sc[2]*sp, ic[0]*sp + (ss[0] + 1) / 2],
                [0, sc[1]*sp, ic[1]*sp + (ss[1] + 1) / 2],
                [0, 0, 1],
            ])

            if not cal['distortion_coefs'][5] == 0:
                print('Unsupported distortion!')
                exit()

            cs.add_camera(cam_matrix,
                          cal['distortion_coefs'][0:5],
                          mc['Rglobal'][i_cal].T,
                          (-mc['Tglobal'][i_cal] @ mc['Rglobal'][i_cal]).reshape(1, 3)
                          )
        return cs
