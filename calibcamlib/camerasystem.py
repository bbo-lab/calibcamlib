import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import yaml
from calibcamlib import Camera
from calibcamlib.helper import intersect, get_line_dist
from calibcamlib.yaml_helper import collection_to_array


# R,t are world->cam
class Camerasystem:
    def __init__(self):
        self.cameras = list()

    def add_camera(self, A, k, rotmat, t, xi=0):
        self.cameras.append({'camera': Camera(A, k, xi=xi), 'R': rotmat, 't': t})

    def project(self, X, offsets=None):
        # Project points in space of shape np.array((..., 3)) to all cameras.
        # Returns image coordinates np.array((N_CAMS, ..., 2))
        if offsets is None:
            offsets = [None for _ in self.cameras]

        X_shape = X.shape
        X = X.reshape(-1, 3)
        x = np.zeros(shape=(len(self.cameras), X.shape[0], 2))

        for i, (c, o) in enumerate(zip(self.cameras, offsets)):
            coords_cam = (c['R'] @ X.T).T + c['t']
            x[i] = c['camera'].space_to_sensor(coords_cam, o).T.T

        return x.reshape((len(self.cameras),) + X_shape[0:-1] + (2,))

    def project_dir(self, V, offsets=None):
        # Project directions in space of shape np.array((..., 3)) to all cameras.
        # Returns image coordinates np.array((N_CAMS, ..., 2))
        # Use this if only a direction from teh camera is known and the camera translation is negligible, e.g.
        #  for objects far away from omnidirectional cameras
        if offsets is None:
            offsets = [None for _ in self.cameras]

        V_shape = V.shape
        V = V.reshape(-1, 3)
        x = np.zeros(shape=(len(self.cameras), V.shape[0], 2))

        for i, (c, o) in enumerate(zip(self.cameras, offsets)):
            coords_cam = (c['R'] @ V.T).T
            x[i] = c['camera'].space_to_sensor(coords_cam, o).T.T

        return x.reshape((len(self.cameras),) + V_shape[0:-1] + (2,))

    def get_camera_lines(self, x, offsets=None):
        # Get camera lines corresponding to image coordinates in shape np.array((N_CAMS, ..., 2)) for all cameras.
        # Returns directions from camera np.array((N_CAMS, ..., 3)) and camera positions np.array((N_CAMS, ..., 3))
        #  in world coordinates (for direct triangulation)
        if offsets is None:
            offsets = [None for _ in self.cameras]

        x_shape = x.shape
        x = x.reshape((x_shape[0], -1, 2))

        V = np.empty(shape=(x.shape[0], x.shape[1], 3))
        P = np.empty(shape=(x.shape[0], x.shape[1], 3))

        for i, o in enumerate(offsets):
            V[i, :], P[i, :] = self.get_camera_lines_cam(x[i], i, o)

        return V.reshape(x_shape[0:-1] + (3,)), P.reshape(x_shape[0:-1] + (3,))

    def get_camera_lines_cam(self, x, cam_idx, offset=None):
        # Get camera lines corresponding to image coordinates in shape np.array((..., 2)) for camera cam_idx.
        # Returns directions from camera np.array((..., 3)) and camera position (tiles to dirs) np.array((..., 3))
        #  in world coordinates.
        # This differes from Camera.sensor_to_space in the translation to world coordinates and the cam pos output
        c = self.cameras[cam_idx]
        V = c['camera'].sensor_to_space(x, offset) @ c['R']
        P = np.tile(- c['t'] @ c['R'], (x.shape[0], 1))

        return V, P

    def triangulate_3derr(self, x, offsets=None):
        # Triangulate image coordinates in shape np.array((N_CAMS, ..., 2)) by finding the closest point to camera lines
        # Returns 3d points np.array((..., 3)) in world coordinates.
        x_shape = x.shape
        x = x.reshape((x_shape[0], -1, 2))

        V, P = self.get_camera_lines(x, offsets)

        X = np.empty(V.shape[1:])
        X[:] = np.NaN

        for i, Xp in enumerate(X):
            if np.sum(~np.isnan(V[:, i, 1])) > 1:
                X[i] = intersect(P[:, i, :], V[:, i, :]).T

        return X.reshape(x_shape[1:-1] + (3,))

    def triangulate_repro(self, x, offsets=None):
        X = self.triangulate_3derr(x, offsets)

        x_shape = x.shape
        x = x.reshape((x_shape[0], -1, 2))
        X = X.reshape(-1, 3)

        for i_point in range(x.shape[1]):
            if np.any(np.isnan(X[i_point])):
                X[i_point] = np.nan
            else:
                res = least_squares(self.repro_error, X[i_point],
                                    method='lm',
                                    verbose=0,
                                    args=[x[:, i_point]],
                                    kwargs={'offsets': offsets, "ravel": True, "nan_to_zero": True})
                X[i_point] = res.x

        return X.reshape(x_shape[1:-1] + (3,))

    def repro_error(self, X, x_orig, offsets=None, ravel=False, nan_to_zero=False):
        err = self.project(X, offsets) - x_orig
        if nan_to_zero:
            err[np.isnan(err)] = 0
        if ravel:
            err = err.ravel()
        return err

    def triangulate(self, x, offsets=None):
        # Triangulate image coordinates in shape np.array((N_CAMS, ..., 2)).
        # Returns 3d points np.array((..., 3)) in world coordinates.
        return self.triangulate_repro(x, offsets)

    def triangulate_nopointcorr(self, AB, offsets=None, linedist_thres=0.2, discard_ambiguities=True, max_points=12):
        # Tries to triangulate image coordinates in shape np.array((N_CAMS, M, 2)) even if image coordinate order does
        # not match between cameras, i.e. AB is shuffled in the second dimension for each camera independently.
        # linedist_thres: distance between camera lines that is still counted as a correspondance
        # max_points: limits number of points on which this is tried
        # Returns 3d points np.array((..., 3)) in world coordinates.
        n_AB = np.array([ab.shape[0] for ab in AB])
        n_cams = len(self.cameras)

        assert n_cams == len(AB), "Number of point sets must match cameras"

        if np.sum(n_AB > 0) < 2:
            return np.zeros((0, 3))

        max_mask = n_AB > max_points
        if np.any(max_mask):
            print(f"Maximum number of points exceeded in cams {np.where(max_mask)[0]}")
            return np.zeros((0, 3))

        cam_bases = np.full((len(AB), 3), np.NaN)

        # Calculate all lines from image coordinates
        full_dirs = []
        for i_cam, (ab, o) in enumerate(zip(AB, offsets)):
            lines, cb = self.get_camera_lines_cam(
                ab,
                i_cam,
                o
            )
            full_dirs.append(lines)
            if len(cb) > 0:
                cam_bases[i_cam] = cb[0]

        correspondences = [np.full((n_cams, n_ab), -1, dtype=int) for n_ab in n_AB]
        AB_sort_idxs = [np.full((n_ab,), -1, dtype=int) for n_ab in n_AB]
        for i_cam in range(n_cams):
            if len(full_dirs[i_cam])==0:
                continue
            # Loop over all following cameras (previous cameras have already been compared the other way round)
            for j_cam in range(i_cam + 1, n_cams):
                if len(full_dirs[j_cam]) == 0:
                    continue
                # Calculate all distances between lines from cam i and j
                distances = np.array(
                    [
                        [
                            get_line_dist(
                                cam_bases[i_cam], full_dirs[i_cam][i, :],
                                cam_bases[j_cam], full_dirs[j_cam][j, :]
                            )
                            for j in range(full_dirs[j_cam].shape[0])
                        ]
                        for i in range(full_dirs[i_cam].shape[0])
                    ]
                )

                # If discard_ambiguities is enabled, ambiguous combinations are discarded
                if discard_ambiguities:
                    connectmat = distances < linedist_thres
                    connectmat[:, np.sum(connectmat, axis=0) > 1] = False  # Discard ambiguities
                # ... else, the smallest distance is used
                else:
                    connectmat = np.all([
                        # Distance is below threshold
                        distances < linedist_thres,
                        # Only select closest line in both directions
                        np.equal(distances, np.min(distances, axis=0, keepdims=True)),
                        np.equal(distances, np.min(distances, axis=1, keepdims=True)),
                    ], axis=0)

                corr_i_cam, corr_j_cam = np.where(connectmat)
                for (corr_i, corr_j) in zip(corr_i_cam, corr_j_cam):
                    correspondences[i_cam][j_cam, corr_i] = corr_j
                    correspondences[j_cam][i_cam, corr_j] = corr_i

        AB_sort = []
        for i_cam in range(n_cams):
            for j_cam in range(i_cam+1, n_cams):
                for i_ab in range(len(correspondences[i_cam][j_cam])):
                    if correspondences[i_cam][j_cam, i_ab] == -1:
                        continue
                    corr_j_idx = correspondences[i_cam][j_cam, i_ab]
                    if AB_sort_idxs[i_cam][i_ab] >= 0:
                        assert (AB_sort_idxs[j_cam][corr_j_idx] == -1 or
                                AB_sort_idxs[j_cam][corr_j_idx] == AB_sort_idxs[i_cam][i_ab]), "Assignment error"
                        AB_sort_idx = AB_sort_idxs[i_cam][i_ab]
                    elif AB_sort_idxs[j_cam][corr_j_idx] >= 0:
                        AB_sort_idx = AB_sort_idxs[j_cam][corr_j_idx]
                    else:
                        AB_sort.append(np.full((n_cams, 2), np.nan))
                        AB_sort_idx = len(AB_sort) - 1
                    AB_sort_idxs[i_cam][i_ab] = AB_sort_idx
                    AB_sort_idxs[j_cam][corr_j_idx] = AB_sort_idx

                    AB_sort[AB_sort_idx][i_cam] = AB[i_cam][i_ab]
                    AB_sort[AB_sort_idx][j_cam] = AB[j_cam][corr_j_idx]

        AB_sort = np.transpose(np.array(AB_sort), (1, 0, 2))
        points = self.triangulate(AB_sort, offsets=offsets)

        points = points[~np.any(np.isnan(points), axis=1)]

        return points

    @staticmethod
    def load(filename: str):
        calibs = Camerasystem.load_dict(filename)
        return Camerasystem.from_calibs(calibs)

    @staticmethod
    def load_dict(filename):
        if filename.endswith('.npy'):
            calibs = np.load(filename, allow_pickle=True)[()]["calibs"]
        elif filename.endswith('.yml'):
            with open(filename, 'r') as stream:
                calibs = yaml.safe_load(stream)["calibs"]
                calibs = collection_to_array(calibs)
        return calibs

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
                          calib['tvec_cam'].reshape(1, 3),
                          xi=calib.get('xi', 0),
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
                [sc[0] * sp, sc[2] * sp, ic[0] * sp + (ss[0] + 1) / 2],
                [0, sc[1] * sp, ic[1] * sp + (ss[1] + 1) / 2],
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
