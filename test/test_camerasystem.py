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
