import unittest
import numpy as np
from calibcamlib.helper import intersect
import pathlib

class TestIntersec(unittest.TestCase):
    def test_intersect(self):
        np.testing.assert_allclose(intersect([[-1,-1],[-1,1]], [[1,1],[1,-1]]), [0,0], atol=1e-10)

        np.testing.assert_allclose(intersect([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 1, 1], [1, 1, -1], [1, -1, 1]]), [0.6, 0.2, 0.4])
