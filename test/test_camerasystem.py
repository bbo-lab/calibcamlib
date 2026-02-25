import unittest
import numpy as np
from calibcamlib import Camerasystem
import pathlib


class TestDistortionFunctions(unittest.TestCase):
    def test_load(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'sample_calibration.yml'

        cs = Camerasystem.load(str(ref_file))

        assert len(cs.cameras) == 4
        assert isinstance(cs.cameras[0]['camera'].A, np.ndarray)
        assert np.isclose(cs.cameras[0]['camera'].A[0, 2], 639.5)

    def test_triangulate_nopointcorr(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'sample_calibration.yml'

        cs = Camerasystem.load(str(ref_file))
        AB = [
            np.array([[0.1, 0.2], [0.4, 0.5], [0.6, 0.7]]),  # Cam 0 - 3 points
            np.array([[0.1, 0.2], [0.4, 0.5]]),  # Cam 1 - 2 points
            np.array([]).reshape(0, 2),  # Cam 2 - no points
            np.array([[0.15, 0.25], [0.45, 0.55]])  # Cam 3 - 2 points
        ]

        # Create dummy offsets (these would usually be camera-specific parameters)
        offsets = [None, None, None, None]

        result = cs.triangulate_nopointcorr(AB, offsets)
        assert result.shape == (0, 3), f"Triangulated points should have shape (0, 3) and not {result.shape}"