import unittest
import numpy as np
from calibcamlib import distortion as dist, Camera
import pathlib


class TestCameraClass(unittest.TestCase):
    def test_space_to_sensor(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'camera_test_space_to_sensor.npy'

        # Build reference file
        # data = {
        #     'points': np.array([[200., 300., 1000.], [-200., 300., 1000.], [200., -300., 1000.], [-200., -300., 1000.]]),
        #     'cams': {
        #         0: {
        #             'A': np.array([
        #                 [1780., 0, 628.3],
        #                 [0, 1780., 505.2],
        #                 [0, 0, 1.],
        #             ]),
        #             'k': np.array([0, 0, 0, 0, 0]),
        #             'xi': 0.91,
        #             'offset': np.array([5, 10])
        #         },
        #     },
        #     "sol": {}
        # }
        #
        # for n, cam in data['cams'].items():
        #     cam = Camera(cam['A'], cam['k'], xi=cam['xi'], offset=cam['offset'])
        #     data['sol'][n] = cam.space_to_sensor(data['points'])
        # np.save(ref_file, data)

        # Test
        data = np.load(ref_file, allow_pickle=True).item()
        for n, cam in data['cams'].items():
            cam = Camera(cam['A'], cam['k'], xi=cam['xi'], offset=cam['offset'])
            np.testing.assert_array_equal(cam.space_to_sensor(data['points']), data['sol'][n])

    def test_sensor_to_space(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'camera_test_sensor_to_space.npy'

        # Build reference file
        # data = {
        #     'points': np.array([[200., 300.], [-200., 300.], [200., -300.], [-200., -300.]]),
        #     'cams': {
        #         0: {
        #             'A': np.array([
        #                 [1780., 0, 628.3],
        #                 [0, 1780., 505.2],
        #                 [0, 0, 1.],
        #             ]),
        #             'k': np.array([0, 0, 0, 0, 0]),
        #             'xi': 0.91,
        #             'offset': np.array([5, 10])
        #         },
        #     },
        #     "sol": {}
        # }
        #
        # for n, cam in data['cams'].items():
        #     cam = Camera(cam['A'], cam['k'], xi=cam['xi'], offset=cam['offset'])
        #     data['sol'][n] = cam.sensor_to_space(data['points'])
        # np.save(ref_file, data)

        # Test
        data = np.load(ref_file, allow_pickle=True).item()
        for n, cam in data['cams'].items():
            cam = Camera(cam['A'], cam['k'], xi=cam['xi'], offset=cam['offset'])
            np.testing.assert_array_equal(cam.sensor_to_space(data['points']), data['sol'][n])

    def test_roundtrip(self):
        ref_file = pathlib.Path(__file__).parent.resolve() / 'data' / 'camera_test_roundtrip.npy'

        # Build reference file
        # data = {
        #     'points': np.array([[200., 300.], [-200., 300.], [200., -300.], [-200., -300.]]),
        #     'cams': {
        #         0: {
        #             'A': np.array([
        #                 [1780., 0, 628.3],
        #                 [0, 1780., 505.2],
        #                 [0, 0, 1.],
        #             ]),
        #             'k': np.array([0, 0, 0, 0, 0]),
        #             'xi': 0.91,
        #             'offset': np.array([5, 10])
        #         },
        #     },
        #     "sol": {}
        # }
        # np.save(ref_file, data)

        # Test
        data = np.load(ref_file, allow_pickle=True).item()
        for n, cam in data['cams'].items():
            cam = Camera(cam['A'], cam['k'], xi=cam['xi'], offset=cam['offset'])
            np.testing.assert_array_almost_equal(cam.space_to_sensor(cam.sensor_to_space(data['points'])), data['points'])


if __name__ == '__main__':
    unittest.main()
