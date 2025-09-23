import multiprocessing
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


class Detections:
    def __init__(self, markers_array=None):
        if markers_array is not None:
            markers_array = deepcopy(markers_array)
            markers_array = self.strip_nans(markers_array)
            markers_array["marker_coords"] = markers_array["marker_coords"].astype(np.float32)
        self._markers_array = markers_array

    @staticmethod
    def strip_nans(markers_array):
        frame_mask = np.any(~np.isnan(markers_array["marker_coords"][..., 0]), axis=(0, 2))
        markers_array["marker_coords"] = markers_array["marker_coords"][:, frame_mask]
        markers_array["detection_idxs"] = np.asarray(markers_array["detection_idxs"])[frame_mask]
        markers_array["frame_idxs"] = np.asarray(markers_array["frame_idxs"])[:, frame_mask]

        marker_mask = np.any(~np.isnan(markers_array["marker_coords"][..., 0]), axis=(0, 1))
        markers_array["marker_coords"] = markers_array["marker_coords"][:, :, marker_mask]
        markers_array["marker_ids"] = np.asarray(markers_array["marker_ids"])[marker_mask]

        assert markers_array["marker_coords"].shape[1] == len(markers_array["detection_idxs"])
        assert markers_array["marker_coords"].shape[2] == len(markers_array["marker_ids"])
        return markers_array

    @staticmethod
    def from_list(markers_list, *args, return_dict=False):
        """
        Converts to array from lists. Also converts a list of corner points with matching frame and marker idx lists
        to a single multidimensional array.
        If the lists are very sparse, this can significantly increase the memory usage, but usually it is not a problem.
        :param markers_list:
        :param args: at least the arg 'marker_ids' is mandatory if 'markers_list' is not a dict
        """
        if isinstance(markers_list, dict):
            marker_coords = markers_list["marker_coords"]
            detection_idxs = markers_list["detection_idxs"]
            frame_idxs = markers_list["frame_idxs"]
            marker_ids = markers_list["marker_ids"]
        else:
            marker_coords = markers_list
            
            expected_args = 3
            # Pad args with None if not enough are provided
            args = args + (None,) * (expected_args - len(args))
            marker_ids = args[0]
            detection_idxs = args[1]
            frame_idxs = args[2]

        if detection_idxs is None:
            detection_idxs = [range(len(marker_coords)) for marker_coords in marker_coords]

        if len(detection_idxs) == 0 or not isinstance(detection_idxs[0], Iterable):
            # Result from single cam
            marker_coords = [marker_coords]
            marker_ids = [marker_ids]
            detection_idxs = [detection_idxs]
            frame_idxs = [frame_idxs]

        n_cams = len(detection_idxs)
        detection_idxs_used = np.unique(np.concatenate(detection_idxs))
        n_frames = len(detection_idxs_used)

        marker_ids_used = np.unique(np.concatenate(sum(marker_ids, [])))
        n_corners = len(marker_ids_used)

        marker_ids_map = {value: idx for idx, value in enumerate(marker_ids_used)}

        marker_coords_array = np.full(shape=(n_cams, n_frames, n_corners, marker_coords[0][0].shape[-1]),
                                      fill_value=np.nan, dtype=np.float32)
        frame_idxs_array = np.full(shape=(n_cams, n_frames), fill_value=-1, dtype=np.int32)

        for i_cam, detection_idxs_cam in enumerate(detection_idxs):
            for i_det, f_idx in enumerate(detection_idxs_used):
                cam_fr_idx = np.where(detection_idxs_cam == f_idx)[0]
                if cam_fr_idx.size < 1:
                    continue
                cam_fr_idx = cam_fr_idx[0]

                indices = np.array(
                    [marker_ids_map[id] for id in np.asarray(marker_ids[i_cam][cam_fr_idx]).ravel()])
                marker_coords_array[i_cam, i_det, indices, :] = marker_coords[i_cam][cam_fr_idx][:, 0]
                if frame_idxs is not None and frame_idxs[i_cam] is not None:
                    frame_idxs_array[i_cam, i_det] = frame_idxs[i_cam][cam_fr_idx]

        markers_array = {
            "marker_coords": marker_coords_array,
            "marker_ids": marker_ids_used,
            "detection_idxs": detection_idxs_used,
            "frame_idxs": frame_idxs_array,
        }

        if return_dict:
            return markers_array
        else:
            return Detections(markers_array)

    @staticmethod
    def from_array(markers_array):
        # TODO: CHeck content
        return Detections(markers_array)

    def to_array(self):
        return deepcopy(self._markers_array)

    def to_list(self):
        """
        Convert to lists compatible to the output of cv2.aruco.detectMarkers() and cv2.aruco.interpolateCornersCharuco()
        but with additional leading camera dimension.
        :return: dictionary of coords, marker ids, detection and frame idxs
        marker_coords: (n_cam,) list of (n_frames,) lists of (n_markers, 1, 2)
        marker_ids: (n_cam,) list of (n_frames,) lists of (n_markers, 1)
        detection_idxs: (n_cam,) list of (n_frames,) lists of (n_markers, ?)
        frame_idxs: (n_cam,) list of (n_frames,) lists of (n_markers, ?)
        """
        mis = self._markers_array["marker_ids"]
        dis = self._markers_array["detection_idxs"]
        fis = self._markers_array["frame_idxs"]
        marker_coords = []
        marker_ids = []
        detection_idxs = []
        frame_idxs = []
        for i_cam, mc_c in enumerate(self._markers_array["marker_coords"]):
            marker_coords_c = []
            marker_ids_c = []
            detection_idxs_c = []
            frame_idxs_c = []
            for detection_idx, frame_idx, mc_f in zip(dis, fis[i_cam], mc_c):
                mask = ~np.isnan(mc_f[:, 0])
                if ~np.any(mask):
                    continue
                marker_coords_c.append(mc_f[mask].reshape(-1, 1, 2))
                frame_idxs_c.append(frame_idx.tolist())
                detection_idxs_c.append(detection_idx.tolist())
                marker_ids_c.append(mis[mask].reshape(-1, 1))
            marker_coords.append(marker_coords_c)
            marker_ids.append(marker_ids_c)
            detection_idxs.append(detection_idxs_c)
            frame_idxs.append(frame_idxs_c)

        return {
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "detection_idxs": detection_idxs,
            "frame_idxs": frame_idxs,
        }

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key,)

        markers_array = {
            "marker_coords": self._markers_array["marker_coords"][key,],
            "marker_ids": self._markers_array["marker_ids"],
            "detection_idxs": self._markers_array["detection_idxs"],
            "frame_idxs": self._markers_array["frame_idxs"][key,],
        }
        markers_array = self.strip_nans(markers_array)

        return Detections(markers_array)

    def __add__(self, o):
        if self._markers_array is None:
            return o

        o_array = o.to_array()
        if not (
                len(self._markers_array["detection_idxs"]) == len(o.to_array()["detection_idxs"]) and
                np.all(self._markers_array["detection_idxs"] == o.to_array()["detection_idxs"]) and
                len(self._markers_array["marker_ids"]) == len(o.to_array()["marker_ids"]) and
                np.all(self._markers_array["marker_ids"] == o.to_array()["marker_ids"])
        ):
            detection_idxs = np.unique(
                np.concatenate((self._markers_array["detection_idxs"], o_array["detection_idxs"])))
            marker_ids = np.unique(np.concatenate((self._markers_array["marker_ids"], o_array["marker_ids"])))
            marker_coords = np.full(
                (
                    len(self._markers_array["marker_coords"]) + len(o_array["marker_coords"]),
                    len(detection_idxs),
                    len(marker_ids),
                    2
                ),
                fill_value=np.nan,
                dtype=self._markers_array["marker_coords"].dtype
            )
            frame_idxs = np.full(marker_coords.shape[:2], fill_value=-1, dtype=int)

            frame_mask = np.isin(detection_idxs, self._markers_array["detection_idxs"])
            marker_mask = np.isin(marker_ids, self._markers_array["marker_ids"])
            len_o1 = len(self._markers_array["marker_coords"])
            for i_cam in range(len_o1):
                marker_coords[i_cam, np.flatnonzero(frame_mask)[:, None], np.flatnonzero(marker_mask)] = \
                    self._markers_array["marker_coords"][i_cam]
                frame_idxs[i_cam, frame_mask] = self._markers_array["frame_idxs"][i_cam]

            frame_mask = np.isin(detection_idxs, o_array["detection_idxs"])
            marker_mask = np.isin(marker_ids, o_array["marker_ids"])
            len_o2 = len(o_array["marker_coords"])
            for i_cam in range(len_o2):
                marker_coords[i_cam + len_o1, np.flatnonzero(frame_mask)[:, None], np.flatnonzero(marker_mask)] = \
                    o_array["marker_coords"][i_cam]
                frame_idxs[i_cam + len_o1, frame_mask] = o_array["frame_idxs"][i_cam]
        else:
            detection_idxs = self._markers_array["detection_idxs"]
            marker_ids = self._markers_array["marker_ids"]
            marker_coords = np.concatenate((self._markers_array["marker_coords"], o_array["marker_coords"]), axis=0)
            frame_idxs = np.concatenate((self._markers_array["frame_idxs"], o_array["frame_idxs"]), axis=0)

        markers_array = {
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "detection_idxs": detection_idxs,
            "frame_idxs": frame_idxs,
        }
        markers_array = self.strip_nans(markers_array)
        return Detections(markers_array)

    def get_frame_detections(self, frame_idx, cam_idxs=None):
        if cam_idxs is None:
            cam_idxs = range(self.get_n_cams())

        if frame_idx in self._markers_array["detection_idxs"]:
            det_idx = self._markers_array["detection_idxs"].tolist().index(frame_idx)
            detections = self._markers_array["marker_coords"][cam_idxs, det_idx]

            return detections
        else:
            return np.full((len(cam_idxs), self.get_n_markers(), self.get_n_dim()), np.nan, dtype=np.float32)

    def is_empty(self):
        if self._markers_array is None:
            return True
        else:
            return False

    def get_n_cams(self):
        """
        Returns camera dimension of contained array
        """
        return self._markers_array["marker_coords"].shape[0]

    def get_n_frames(self):
        """
        Returns frames dimension of contained array
        """
        return self._markers_array["marker_coords"].shape[1]

    def get_n_markers(self):
        """
        Returns marker dimension of contained array
        """
        return self._markers_array["marker_coords"].shape[2]

    def get_n_dim(self):
        """
        Returns marker coord dimension of contained array
        """
        return self._markers_array["marker_coords"].shape[3]

    def get_n_detections(self):
        """
        Returns number of overall detections
        """
        return np.sum(~np.isnan(self._markers_array["marker_coords"][:, :, :, 1]))

    def get_n_detections_frames(self):
        """
        Returns number of detected frames per marker (shape cam x markers)
        """
        return np.sum(~np.isnan(self._markers_array["marker_coords"][:, :, :, 1]), axis=1)

    def get_n_detections_markers(self):
        """
        Returns number of detected markers per frame (shape cam x frames)
        """
        return np.sum(~np.isnan(self._markers_array["marker_coords"][:, :, :, 1]), axis=2)

    @staticmethod
    def from_file(detection_files):
        if isinstance(detection_files, list):
            with multiprocessing.Pool() as pool:
                detections_list = pool.map(Detections.from_file, detection_files)
            return sum(detections_list, Detections())

        detection_files = Path(detection_files)
        if detection_files.suffix == ".yml":
            with open(detection_files, "r") as file:
                try:
                    from yaml import CSafeLoader
                    detection = yaml.load(file, Loader=CSafeLoader)
                except ImportError:
                    detection = yaml.safe_load(file)
        elif detection_files.suffix == ".npy":
            detection = np.load(detection_files, allow_pickle=True)[()]
        else:
            raise FileNotFoundError(f"{detection_files} is not supported")

        if "marker_coords" in detection:
            marker_coords = np.array(detection["marker_coords"], dtype=np.float32)
        elif "corners" in detection:
            marker_coords = np.array([detection["corners"]], dtype=np.float32)
        else:
            # TODO: write import code for multicamcal files
            raise ValueError("Unsupported dictionary content")

        if "marker_ids" in detection:
            marker_ids = np.array(detection["marker_ids"])
        elif "used_corner_ids" in detection:
            marker_ids = np.array(detection["used_corner_ids"])
        else:
            marker_ids = np.arange(np.array(marker_coords).shape[1])

        if "detection_idxs" in detection:
            detection_idxs = np.array(detection["detection_idxs"])
        elif "used_frames_ids" in detection:
            detection_idxs = np.array(detection["used_frames_ids"])
        else:
            detection_idxs = np.arange(np.array(marker_coords).shape[0])

        if "frame_idxs" in detection:
            frame_idxs = np.array(detection["frame_idxs"])
        elif "used_frames_ids" in detection:
            frame_idxs = np.array(detection["used_frames_ids"])
        else:
            frame_idxs = np.arange(np.array(marker_coords).shape[0])

        return Detections({
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "detection_idxs": detection_idxs,
            "frame_idxs": frame_idxs,
        })

    def to_file(self, file_paths):
        if isinstance(file_paths, str):
            file_paths = Path(file_paths)

        if isinstance(file_paths, Path):
            file_paths = [file_paths.parent / f"{file_paths.stem}_{i:03d}{file_paths.suffix}"
                          for i in range(len(self._markers_array["marker_coords"]))]

        assert len(file_paths) == len(
            self._markers_array["marker_coords"]), "Number of files must match number of detections"

        for i_cam, file_path in enumerate(file_paths):
            markers_dict = {
                "version": "2.0",
                "storage_method": "array",
                "marker_coords": self._markers_array["marker_coords"][(i_cam,),].tolist(),
                "marker_ids": self._markers_array["marker_ids"].tolist(),
                "detection_idxs": self._markers_array["detection_idxs"].tolist(),
                "frame_idxs": self._markers_array["frame_idxs"][(i_cam,),].tolist(),

            }
            if Path(file_path).suffix == ".yml":
                with open(file_path, "w") as file:
                    yaml.safe_dump(markers_dict, file)
            elif Path(file_path).suffix == ".npy":
                np.save(file_path, markers_dict)
            else:
                raise FileNotFoundError(f"{file_path} is not supported")
