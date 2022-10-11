import unittest
import numpy as np
from calibcamlib import distortion as dist
import pathlib


class TestDistortionFunctions(unittest.TestCase):
    def test_distort(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'distortion_test_distort.npy'

        # # Build reference file
        # data = {
        #     'points': np.array([[2000., 1000.], [-300., 200.], [300., -1200.], [-200., 300.]]),
        #     'k': {
        #         0: np.array([0, 0, 0, 0, 0]),
        #         1: np.array([0.1, 0.1, 0, 0, 0]),
        #         2: np.array([-0.1, 0.1, 0, 0, 0]),
        #         3: np.array([0.1, -0.1, 0, 0, 0]),
        #         4: np.array([-0.1, -0.1, 0, 0, 0]),
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
        return
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'distortion_test_distort_inverse.npy'

        # # Build reference file
        # data = {
        #     'points': np.array([[2000., 1000.], [-300., 200.], [300., -1200.], [-200., 300.]]),
        #     'k': {
        #         0: np.array([0, 0, 0, 0, 0]),
        #         1: np.array([0.1, 0.1, 0, 0, 0]),
        #         2: np.array([-0.1, 0.1, 0, 0, 0]),
        #         3: np.array([0.1, -0.1, 0, 0, 0]),
        #         4: np.array([-0.1, -0.1, 0, 0, 0]),
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
            np.testing.assert_array_equal(dist.distort_inverse(data['points'], ks), data['sol'][n])

    def test_distort_roundtrip(self):
        return
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'distortion_test_distort_roundtrip.npy'

        # # Build reference file
        # data = {
        #     'points': np.array([[2000., 1000.], [-300., 200.], [300., -1200.], [-200., 300.]]),
        #     'k': {
        #         0: np.array([0, 0, 0, 0, 0]),
        #         1: np.array([0.1, 0.1, 0, 0, 0]),
        #     },
        #     "sol": {}
        # }
        # np.save(ref_file, data)

        # Test
        data = np.load(ref_file, allow_pickle=True).item()
        for n, ks in data['k'].items():
            np.testing.assert_array_almost_equal(dist.distort(dist.distort_inverse(data['points'], ks), ks), data['points'])
            np.testing.assert_array_almost_equal(dist.distort_inverse(dist.distort(data['points'], ks), ks), data['points'])
            pass


if __name__ == '__main__':
    unittest.main()
