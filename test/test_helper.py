import unittest
import numpy as np
from calibcamlib.helper import intersect

import pathlib

class TestIntersec(unittest.TestCase):
    def test_intersect(self):
        np.testing.assert_allclose(intersect([[-1,-1],[-1,1]], [[1,1],[1,-1]]), [0,0], atol=1e-10)

        np.testing.assert_allclose(intersect([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 1, 1], [1, 1, -1], [1, -1, 1]]), [0.6, 0.2, 0.4])

    def test_intersect_vectorized_corrected(self):
        bases_test = np.random.rand(5, 3, 3)  # 5 sets of 3 lines in 3D space
        vecs_test = np.random.rand(5, 3, 3)

        results_original = np.array([
            intersect(bases_test[i], vecs_test[i]) for i in range(bases_test.shape[0])
        ])

        results_vectorized = intersect(bases_test, vecs_test)

        assert np.allclose(results_original, results_vectorized, equal_nan=True), "Test failed!"

        return "Test passed!"