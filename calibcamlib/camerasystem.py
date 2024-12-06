import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import yaml
from calibcamlib import Camera
from calibcamlib.helper import intersect, get_line_dist
from calibcamlib.yaml_helper import collection_to_array
from collections.abc import Iterable
from bbo.geometry import Line


# R,t are world->cam
class Camerasystem:
    def __init__(self, unit=None):
        self.cameras = list()
        self.unit = unit

    def add_camera(self, A, k, rotmat, t, xi=0):
        self.cameras.append({'camera': Camera(A, k, xi=xi), 'R': rotmat, 't': t})

    def project(self, X, offsets=None, cam_idx=None):
        # Project points in space of shape np.array((..., 3)) to all cameras.
        # Returns image coordinates np.array((N_CAMS, ..., 2))

        X_shape = X.shape
        assert X_shape[-1] == 3, f"Vectors have to be in 3d-space, last dimension was {X_shape[-1]}"
        X = X.reshape(-1, 3)

        if cam_idx is None:
            cam_idx = self.cameras
            cam_shape = (len(cam_idx),)
        elif isinstance(cam_idx, Iterable):
            cam_idx = [self.cameras[idx] for idx in cam_idx]
            cam_shape = (len(cam_idx),)
        else:
            cam_idx = [self.cameras[cam_idx]]
            cam_shape = ()
        x = np.zeros(shape=(len(cam_idx), X.shape[0], 2), dtype=float)

        if offsets is None:
            offsets = np.full(shape=len(cam_idx), fill_value = None)

        for i, (c, o) in enumerate(zip(cam_idx, offsets)):
            X_cam = self.camsystem_to_cam(X, i)
            x[i] = c['camera'].space_to_sensor(X_cam, o)

        return x.reshape((*cam_shape, *X_shape[0:-1],2))

    def camsystem_to_cam(self, X_world, cam_idx, normalize=False):
        X_world_shape = X_world.shape

        # TODO: Rewrite this to just keep dimensions
        if len(X_world_shape) != 2:
            assert X_world_shape[-1] == 3, "Input vector must have dimension 3"
            X_world = X_world.reshape(-1, 3)

        X_cam = X_world @ self.cameras[cam_idx]['R'].T + self.cameras[cam_idx]['t']

        if normalize:
            X_cam /= np.linalg.norm(X_cam, axis=-1, keepdims=True)

        if len(X_world_shape) != 2:
            X_cam = X_cam.reshape(X_world_shape)

        return X_cam

    def get_camsystem_to_cam_transformation(self, cam_idx):
        from bbo.geometry import RigidTransform
        return RigidTransform(rotation=self.cameras[cam_idx]['R'], translation=self.cameras[cam_idx]['t'],
                              rotation_type="matrix")

    def get_camera_positions(self):
        return np.array([- c['t'] @ c['R'] for c in self.cameras])

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
            coords_cam = V @ c['R'].T
            x[i] = c['camera'].space_to_sensor(coords_cam, o)

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

    def get_camera_lines_cam(self, x, cam_idx, offset=None, use_geometry=False):
        # Get camera lines corresponding to image coordinates in shape np.array((..., 2)) for camera cam_idx.
        # Returns directions from camera np.array((..., 3)) and camera position (tiles to dirs) np.array((..., 3))
        #  in world coordinates.
        # This differes from Camera.sensor_to_space in the translation to world coordinates and the cam pos output
        x_shape = x.shape
        x = x.reshape((x_shape[0], -1, 2))
        #TODO documentation is incorrect, does only allow flattened arrays
        c = self.cameras[cam_idx]
        V = c['camera'].sensor_to_space(x, offset) @ c['R']
        P = np.tile(- c['t'] @ c['R'], (x.shape[0], 1))
        V, P = V.reshape(x_shape[0:-1] + (3,)), P.reshape(x_shape[0:-1] + (3,))
        if use_geometry:
            return Line(position=P, direction=V)
        return V, P

    def triangulate_3derr(self, x, offsets=None):
        # Triangulate image coordinates in shape np.array((N_CAMS, ..., 2)) by finding the closest point to camera lines
        # Returns 3d points np.array((..., 3)) in world coordinates.
        x_shape = x.shape
        x = x.reshape((x_shape[0], -1, 2))

        V, P = self.get_camera_lines(x, offsets)

        X = np.full(V.shape[1:], np.nan)

        for i, Xp in enumerate(X):
            if np.count_nonzero(~np.isnan(V[:, i, 1])) > 1:
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
        if np.sum(n_AB > 0) < 2:
            return np.zeros((0, 3))

        max_mask = n_AB <= max_points
        if not np.any(max_mask):
            return np.zeros((0, 3))
        main_cam_idx = np.where(n_AB == np.max(n_AB[max_mask]))[0][0]
        if n_AB[main_cam_idx] == 0:
            return np.zeros((0, 3))

        cam_bases = np.empty((len(AB), 3))
        cam_bases[:] = np.nan

        full_ab = np.empty((len(AB), n_AB[main_cam_idx], 2))
        full_ab[:] = np.nan
        full_ab[main_cam_idx] = AB[main_cam_idx]

        full_dirs = np.empty((len(AB), n_AB[main_cam_idx], 3))
        full_dirs[:] = np.nan
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

            connectmat = np.all([distances < linedist_thres,
                                 np.equal(distances, np.min(distances, axis=1)[:, np.newaxis]),
                                 # np.equal(distances,np.min(distances,axis=0)[np.newaxis,:])
                                 ], axis=0)
            if discard_ambiguities:
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

        return points

    @staticmethod
    def load(filename: str | Path, unit=None):
        filename = Path(filename)
        calibration = Camerasystem.load_dict(filename)
        calibration = Camerasystem.adjust_calibration_unit(calibration, unit)
        return Camerasystem.from_calibs(calibration["calibs"],
                                        unit=calibration["board_params"]["unit"])

    @staticmethod
    def adjust_calibration_unit(calibration, unit):
        if "board_params" not in calibration:
            # warnings.warn("Unit of calibration could not be determined! Returning as is. Avoid this message by adding a"
            #               " unit in calibration['board_params']['unit'].")
            calibration["board_params"] = {"unit": None}
            return calibration
        elif "unit" not in calibration["board_params"]:
            # warnings.warn("Unit of calibration could not be determined! Returning as is. Avoid this message by adding a"
            #               " unit in calibration['board_params']['unit'].")
            calibration["board_params"]["unit"] = None
            return calibration
        elif unit is None:
            return calibration


        factor_dict = {"m": 1, "cm": 0.01, "mm": 0.001}

        factor = factor_dict[calibration["board_params"]["unit"]]/factor_dict[unit]
        calibration["board_params"]["unit"] = unit
        for calib in calibration["calibs"]:
            calib["tvec_cam"] *= factor
        for key in calibration["board_params"]:
            if key.endswith("_real"):
                calibration["board_params"][key] *= factor
        if "info" in calibration:
            calibration["info"]["tvecs_boards"] = (np.asarray(calibration["info"]["tvecs_boards"]) * factor).tolist()

        return calibration

    @staticmethod
    def load_dict(filename: str | Path):
        filename = Path(filename)
        if filename.suffix == '.npy':
            calibration = np.load(filename.as_posix(), allow_pickle=True)[()]
        elif filename.suffix == '.yml':
            with open(filename, 'r') as stream:
                calibration = yaml.safe_load(stream)
                calibration["calibs"] = collection_to_array(calibration["calibs"])
        return calibration

    @staticmethod
    def from_calibcam_file(filename: str | Path):
        filename = Path(filename)

        cs = Camerasystem()
        calib = np.load(filename.as_posix(), allow_pickle=True)[()]

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
    def from_calibs(calibs, unit=None):
        cs = Camerasystem(unit=unit)

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


def strip_calibs(calibs):
    # Strips the calibration down to the minimal info necessary by calibcamlib and calibcam
    calibs_new = []
    parameters = ["A", "k", "xi", "rvec_cam", "tvec_cam"]
    for c in calibs:
        calib_new = {}
        for param in parameters:
            calib_new[param] = c[param]
        calibs_new.append(calib_new)
    return calibs_new
