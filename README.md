# calibcamlib
Library to work with calibration from bbo-calibcam

## Installation

Install via pip as `bbo-calibcamlib`.

## Usage

Load a calibration with

```python
cs = calibcamlib.Camerasystem.load([PATH TO multicam_calibration.yml])
```

Use calibrated camera system with

```python
coords2d = cs.project(self, X, offsets=None)
        # Project points in space of shape np.array((..., 3)) to all cameras.
        # Returns image coordinates np.array((N_CAMS, ..., 2))

dirs, cam_pos = cs.get_camera_lines(self, x, offsets=None)
        # Get camera lines corresponding to image coordinates in shape np.array((N_CAMS, ..., 2)) for all cameras.
        # Returns directions from camera np.array((N_CAMS, ..., 3)) and camera positions np.array((N_CAMS, ..., 3))
        #  in world coordinates (for direct triangulation)

coords3d = cs.triangulate(self, x, offsets=None)
        # Triangulate image coordinates in shape np.array((N_CAMS, ..., 2)) by minimizing reprojection error.
        # Returns 3d points np.array((..., 3)) in world coordinates.

coords3d = cs.triangulate_3derr(self, x, offsets=None)
        # Triangulate image coordinates in shape np.array((N_CAMS, ..., 2)) by finding the closest point to camera lines
        # Returns 3d points np.array((..., 3)) in world coordinates.
```