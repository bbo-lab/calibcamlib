import numpy as np
import pyvista as pv

from calibcamlib import Camerasystem


class PyvistaCalibrationPlotter:
    """
    PyVista visualization of a calibrated camera system.

    The plotter shows:
      - the camera windows / image planes
      - the camera optical axes
      - optionally, the current video frame on each camera window

    The calibration plot itself is static. Video frames can be changed with
    ``set_frame(iframe)``.

    Parameters
    ----------
    camerasystem
        Camera system providing ``cameras`` and ``get_camera_lines(...)``.

    plotter
        A ``pyvista.Plotter`` instance.

    transformation
        Sequence of transformations, one per video frame. A transformation
        must provide::

            transformation[iframe].apply(points, only_linear=True/False)

        ``only_linear=False`` is used for absolute positions.
        ``only_linear=True`` is used for vectors/directions.

    videoreaders, optional
        Video readers. This can be a mapping from video key to reader or a
        sequence corresponding to ``camerasystem.cameras``.

        A reader is expected to provide a frame through ``get_frame(iframe)``.
    """

    def __init__(
        self,
        camerasystem,
        plotter,
        transformation=None,
        videoreaders=None,
        scale=0.12,
        color = None,
        subsurf_camera_window=6,
        wide_angle_camera=False,
        default_camerasize=None
    ):
        self.camerasystem = camerasystem
        self.plotter = plotter
        self.transformation = transformation
        self.videoreaders = videoreaders
        self.scale = scale
        self.color = color
        self.subsurf_camera_window = subsurf_camera_window
        self.actors = []
        self.num_cameras = len(camerasystem.cameras)
        self.wide_angle_camera = wide_angle_camera
        if default_camerasize is not None:
            self.default_height, self.default_width = default_camerasize
        else:
            self.default_height, self.default_width = 1080, 1920
        self.setup_cameras()

    def get_camera_shape(self, icam):
        camera = self.camerasystem.cameras[icam]
        return camera.get("height", self.default_height), camera.get("width", self.default_width)

    def setup_cameras(self):
        self.actors = []
        for icam in range(self.num_cameras):
            height, width = self.get_camera_shape(icam)

            pixel_coords = np.stack(
                np.meshgrid(
                    np.linspace(0, width - 1, self.subsurf_camera_window),
                    np.linspace(0, height - 1, self.subsurf_camera_window),
                    indexing="xy",
                ),
                axis=-1,
            ).reshape(-1, 2)

            c = self.camerasystem.cameras[icam]
            current_camera_lines = c['camera'].sensor_to_space(pixel_coords.reshape(-1, 2), None)

            if not self.wide_angle_camera:
                # Normalize camera_lines_direction
                current_camera_lines /= current_camera_lines[:, 2:3]

            camera_lines = current_camera_lines @ c['R']

            current_camera_lines = np.concatenate((
                np.zeros(shape=(1,3)),
                camera_lines.reshape(-1, 3)), axis=0)

            current_camera_lines *= self.scale
            current_camera_lines -= c['t'] @ c['R']

            vertices = current_camera_lines.reshape(-1, 3)

            outer_vertices = 1 + np.array([
                0,
                self.subsurf_camera_window - 1,
                self.subsurf_camera_window * (self.subsurf_camera_window - 1),
                self.subsurf_camera_window * self.subsurf_camera_window - 1
            ])

            n = self.subsurf_camera_window

            outer_loop_vertices = np.concatenate([
                np.arange(1, n + 1),  # top
                np.arange(2 * n, n * n + 1, n),  # right
                np.arange(n * n - 1, n * (n - 1), -1),  # bottom
                np.arange(n * (n - 2) + 1, 1, -n),  # left
            ])

            center_corner_edges = np.stack(
                (
                    np.full(shape=4, fill_value=2, dtype=int),
                    np.zeros(shape=4, dtype=int),
                 outer_vertices), axis=-1).ravel()

            outer_edge_edges = np.concatenate(
                (
                    [len(outer_loop_vertices) + 1],
                    outer_loop_vertices,
                    [outer_loop_vertices[0]]
                ))

            vertices_left_up = (np.arange(0, self.subsurf_camera_window - 1)[np.newaxis, :]
                                + self.subsurf_camera_window * np.arange(0, self.subsurf_camera_window - 1)[:, np.newaxis] + 1)
            vertices_left_up = vertices_left_up.ravel()

            surface_faces = np.stack((
                np.full(shape=len(vertices_left_up), fill_value=4, dtype=int),
                vertices_left_up,
                vertices_left_up + 1,
                vertices_left_up + self.subsurf_camera_window + 1,
                vertices_left_up + self.subsurf_camera_window
            ),axis=-1)

            lines = np.concatenate((center_corner_edges, outer_edge_edges))

            poly_data = pv.PolyData(vertices, faces=surface_faces, lines=lines)
            self.actors.append(self.plotter.add_mesh(
                poly_data,
                color=self.color,
                opacity=0.3,
                show_edges=True,
                line_width=5
            ))


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame(self, iframe):
        if self.transformation is not None:
            current_transformation = self.transformation[iframe]
            transformation_matrix = current_transformation.as_matrix()
            for actor in self.actors:
                actor.SetUserMatrix(transformation_matrix)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PyVista calibration plotter")
    parser.add_argument("--input", nargs="+", required=True, help="Input video files")
    parser.add_argument("--output", required=False, help="Output file for the plot")
    parser.add_argument("--camerasize", type=int, default=None, nargs=2)
    parser.add_argument("--wide-angle", action="store_true", help="Use wide angle camera model")
    args = parser.parse_args()

    # Cretes a pyvista scene with all the calibrations
    plotter = pv.Plotter()
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    for iinput, input_file in enumerate(args.input):
        multicalibration = Camerasystem.load(input_file)
        #set up pyvista plotter
        pcp = PyvistaCalibrationPlotter(
            camerasystem=multicalibration,
            plotter=plotter,
            color=colors[iinput],
            wide_angle_camera=args.wide_angle,
            default_camerasize=args.camerasize)

    if args.output is None:
        plotter.show()