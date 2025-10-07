import os
from collections.abc import Iterable
from pathlib import Path

from scipy.spatial.transform import Rotation as R
import numpy as np


class Board:
    def __init__(self, board_params: dict):
        self.board_params = board_params

    @staticmethod
    def from_file(board_path, board_idx=None):
        if not isinstance(board_path, str) and isinstance(board_path, Iterable):
            boards_list = [Board.from_file(bp) for bp in board_path]
            if board_idx is not None:
                return boards_list[board_idx]
            else:
                return boards_list

        board_path = Path(board_path)
        if board_path.is_file():
            board_path = board_path.as_posix()
        elif board_path.is_dir():
            board_path = board_path / 'board.npy'
        else:
            board_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../boards',
                                      board_path.as_posix() + '.npy')

        board_params = np.load(os.path.expanduser(board_path), allow_pickle=True).item()

        if board_params is not None:
            board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']  # noqa

        return Board(board_params)

    def get_board_params(self):
        return self.board_params

    def get_cv2_board(self, zero_ids=False):
        import cv2
        from cv2 import aruco
        board_params = self.board_params
        
        ids = board_params.get('ids', None)
        if ids is not None and zero_ids:
            ids = ids - ids[0]

        try:
            board = cv2.aruco.CharucoBoard((board_params['boardWidth'],
                                       board_params['boardHeight']),
                                       board_params['square_size_real'],
                                       board_params['marker_size'] * board_params['square_size_real'],
                                       cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']),
                                       ids=ids)
        except Exception as e:
            print(e)
            raise e

        if "legacy" in board_params:
            board.setLegacyPattern(board_params["legacy"])
        elif "version" not in board_params:  # TODO: This identification might still need some refinement
            board.setLegacyPattern(True)

        return board

    def get_board_points(self, exact=False):
        board_params = self.board_params

        board_width = board_params['boardWidth']
        board_height = board_params['boardHeight']
        if exact:
            square_size_x = board_params['square_size_real_y']
            square_size_y = board_params['square_size_real_x']
        else:
            square_size_x = board_params['square_size_real']
            square_size_y = board_params['square_size_real']

        n_corners = (board_width - 1) * (board_height - 1)

        board_0 = np.repeat(np.arange(1, board_width).reshape(1, board_width - 1), board_height - 1,
                            axis=0).ravel().reshape(n_corners, 1)
        board_1 = np.repeat(np.arange(1, board_height), board_width - 1, axis=0).reshape(n_corners, 1)
        board_2 = np.zeros(n_corners).reshape(n_corners, 1)
        board_points = np.concatenate([board_0 * square_size_x, board_1 * square_size_y,
                                       board_2], 1)

        if "rotation" in board_params:
            board_points = R.from_rotvec(board_params["rotation"]).apply(board_points)
        if "offset" in board_params:
            board_points = board_points + np.array(board_params["offset"]).reshape(1, 3)

        return board_points  # n_corners x 3

    def get_board_ids(self):
        # Returns ARUCO ids of boards (these are different from charuco corner ids!)
        if "ids" in self.board_params:
            return self.board_params["ids"]
        else:
            return np.arange((self.board_params["boardWidth"]*self.board_params["boardHeight"]) // 2)

    def get_corner_ids(self, zero_ids=False):
        # Returns corner ids. openCV starts corner ids at 0 regardless of aruco patterns. In calibcam convention,
        # corner ids start at the smallest aruco id.
        # Caveat: openCV functions that rely on the openCV board object like calibrateCameraCharucoExtended
        # require corner ids start at 0.
        ids = np.arange(
            (self.board_params["boardWidth"] - 1) * (self.board_params["boardHeight"] - 1)
        )
        if not zero_ids:
            ids = ids + self.get_board_ids()[0]
        return ids

    def get_board_img(self, pixel_size=None):
        if pixel_size is None:
            board_params = self.get_board_params()
            marker_ratio = board_params["marker_size"]
            rows = board_params["boardWidth"]
            columns = board_params["boardHeight"]
            aruco_dict = board_params["dictionary_type"]

            match aruco_dict:
                case aruco.DICT_4X4_250:
                    aruco_size = 4
                case aruco.DICT_5X5_250:
                    aruco_size = 5
                case _:
                    raise NotImplementedError

            # Generate the Charuco board image
            pixel_size = (round(((aruco_size + 2) / marker_ratio) * rows),
                          round(((aruco_size + 2) / marker_ratio) * columns))

        board = self.get_cv2_board()
        return board.generateImage(pixel_size)
