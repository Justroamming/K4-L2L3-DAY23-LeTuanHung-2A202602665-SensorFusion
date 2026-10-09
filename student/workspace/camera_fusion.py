"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray
MIN_CAMERA_DEPTH = 1e-6


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    # Transform position from vehicle to sensor frame: p_s = R @ p + t
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    transform = np.asarray(sensor.veh_to_sens)
    p_s = transform[:3, :3] @ position + transform[:3, 3]

    # Check for non-finite coordinates
    if not np.isfinite(p_s).all():
        return False

    # Camera requires positive depth
    if sensor.name == "camera" and p_s[0] <= MIN_CAMERA_DEPTH:
        return False

    # Check horizontal angle (atan2(y_s, x_s)) within FOV
    angle = np.arctan2(p_s[1], p_s[0])
    fov_min, fov_max = sensor.fov
    return bool(fov_min <= angle <= fov_max)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    # Transform position from vehicle to sensor frame
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    transform = np.asarray(sensor.veh_to_sens)
    p_s = transform[:3, :3] @ position + transform[:3, 3]

    # Check for non-finite coordinates and positive depth
    if not np.isfinite(p_s).all() or p_s[0] <= MIN_CAMERA_DEPTH:
        raise ValueError(
            f"Camera projection needs finite coordinates and positive depth "
            f"> {MIN_CAMERA_DEPTH:g}; sensor position={p_s.tolist()}"
        )

    # Pinhole projection: u = c_i - f_i * y_s/x_s; v = c_j - f_j * z_s/x_s
    depth, left, up = p_s[0], p_s[1], p_s[2]
    u = sensor.c_i - sensor.f_i * left / depth
    v = sensor.c_j - sensor.f_j * up / depth

    return np.asmatrix([[u], [v]], dtype=float)


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    z_mat = np.asmatrix([[z[0]], [z[1]]], dtype=float)
    R = np.asmatrix(np.diag(np.array([params.sigma_cam_i**2, params.sigma_cam_j**2], dtype=float)))
    return {"z": z_mat, "R": R, "sensor": sensor}