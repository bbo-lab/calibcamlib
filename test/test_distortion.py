import unittest
import numpy as np
from calibcamlib import distortion as dist
import pathlib


class TestDistortionFunctions(unittest.TestCase):
    def test_distort(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'distortion_test_distort.npy'

        # # Build reference file
        # data = {
        #     'points': np.array([[-0.23780899, -0.10966292],
        #                         [-0.46252809, -0.10966292],
        #                         [-0.23780899, -0.44674157],
        #                         [-0.46252809, -0.44674157]]),
        #     'k': {
        #         0: np.array([0, 0, 0, 0, 0]),
        #         1: np.array([0.29766065, 0.0482777, 0, 0, 0]),
        #         2: np.array([-0.29766065, 0.0482777, 0, 0, 0]),
        #         3: np.array([0.29766065, -0.0482777, 0, 0, 0]),
        #         4: np.array([-0.29766065, -0.0482777, 0, 0, 0]),
        #         5: np.array([-0.29766065, 0.0482777, 0.00087529, 0.00336191, 0.0001]),
        #         6: np.array([-0.3, 0.1, 0.2, 0.4, 0.15]),
        #     },
        #     "sol": {}
        # }
        #
        # for n, ks in data['k'].items():
        #     data['sol'][n] = dist.distort(data['points'], ks)
        # np.save(ref_file, data)

        # Test
        data = np.load(ref_file, allow_pickle=True).item()
        for n, ks in data['k'].items():
            np.testing.assert_array_equal(dist.distort(data['points'], ks), data['sol'][n])

    def test_distort_inverse(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'distortion_test_distort_inverse.npy'

        # # Build reference file - ALWAYS RUN TEST BEFORE RERUNNING THIS!!!
        # data = {
        #     'points': np.array([[-0.23780899, -0.10966292],
        #                         [-0.46252809, -0.10966292],
        #                         [-0.23780899, -0.44674157],
        #                         [-0.46252809, -0.44674157]]),
        #     'k': {
        #         0: np.array([0, 0, 0, 0, 0]),
        #         1: np.array([0.29766065, 0.0482777, 0, 0, 0]),
        #         2: np.array([-0.29766065, 0.0482777, 0, 0, 0]),
        #         3: np.array([0.29766065, -0.0482777, 0, 0, 0]),
        #         4: np.array([-0.29766065, -0.0482777, 0, 0, 0]),
        #         5: np.array([-0.29766065, 0.0482777, 0.00087529, 0.00336191, 0.0001]),
        #     },
        #     "sol": {}
        # }
        #
        # for n, ks in data['k'].items():
        #     data['sol'][n] = dist.distort_inverse(data['points'], ks)
        # np.save(ref_file, data)

        # Test
        data = np.load(ref_file, allow_pickle=True).item()
        for n, ks in data['k'].items():
            np.testing.assert_array_almost_equal(dist.distort_inverse(data['points'], ks), data['sol'][n])

    def test_distort_roundtrip(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'distortion_test_distort_roundtrip.npy'

        # # Build reference file
        # data = {
        #     'points': np.array([[-0.23780899, -0.10966292],
        #                         [-0.46252809, -0.10966292],
        #                         [-0.23780899, -0.44674157],
        #                         [-0.46252809, -0.44674157]]),
        #     'k': {
        #         0: np.array([0, 0, 0, 0, 0]),
        #         1: np.array([0.29766065, 0.0482777, 0, 0, 0]),
        #         2: np.array([-0.29766065, 0.0482777, 0, 0, 0]),
        #         3: np.array([0.29766065, -0.0482777, 0, 0, 0]),
        #         4: np.array([-0.29766065, -0.0482777, 0, 0, 0]),
        #         5: np.array([-0.29766065, 0.0482777, 0.00087529, 0.00336191, 0.0001]),
        #     },
        #     "sol": {}
        # }
        # np.save(ref_file, data)

        # Test
        data = np.load(ref_file, allow_pickle=True).item()
        for n, ks in data['k'].items():
            np.testing.assert_array_almost_equal(dist.distort(dist.distort_inverse(data['points'], ks), ks),
                                                 data['points'])
            np.testing.assert_array_almost_equal(dist.distort_inverse(dist.distort(data['points'], ks), ks),
                                                 data['points'])

    def test_dist_opt_func(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'distortion_test_distort.npy'

        data = np.load(ref_file, allow_pickle=True).item()
        for n, ks in data['k'].items():
            for p, p_d in zip(data['points'], data['sol'][n]):
                np.testing.assert_array_almost_equal(dist.dist_opt_func(p, p_d, ks), np.zeros_like(p))


if __name__ == '__main__':
    unittest.main()
