"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()

    # Transform measurement position from sensor to vehicle frame
    # meas.z is in sensor frame (3x1), need to transform using sensor's sens_to_veh
    z_sensor = np.asarray(meas.z, dtype=float).reshape(-1)[:3]  # [x, y, z] in sensor frame
    transform = np.asarray(meas.sensor.sens_to_veh)
    z_vehicle = transform[:3, :3] @ z_sensor + transform[:3, 3]

    # State vector: position + zero velocity
    x = np.asmatrix(np.r_[z_vehicle, [0.0, 0.0, 0.0]]).T

    # Covariance matrix
    # Position block from rotated R
    R_rot = transform[:3, :3] @ np.asarray(meas.R) @ transform[:3, :3].T
    # Velocity block from sigma_p44, sigma_p55, sigma_p66
    P_vel = np.diag([params.sigma_p44**2, params.sigma_p55**2, params.sigma_p66**2])

    P = np.asmatrix(np.block([
        [R_rot, np.zeros((3, 3))],
        [np.zeros((3, 3)), P_vel]
    ]))

    # Initial score and state
    score = 1.0 / params.window
    state = "initialized"

    return {"x": x, "P": P, "state": state, "score": score}


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()

    score = track["score"]
    state = track["state"]

    if associated:
        # Hit: add 1/window, cap at 1
        score = min(score + 1.0 / params.window, 1.0)
    else:
        # Miss in FOV: subtract 1/window
        score = score - 1.0 / params.window

    # State transitions
    if state == "confirmed":
        # Confirmed tracks stay confirmed even after misses
        pass
    elif score > params.confirmed_threshold:
        state = "confirmed"
    elif associated:
        # Hit but not yet confirmed
        state = "tentative"
    else:
        # Miss and not confirmed
        state = "tentative"

    track["score"] = score
    track["state"] = state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()

    P = np.asarray(track["P"])
    score = track["score"]
    state = track["state"]

    # Check horizontal position variance (P[0,0] = Pxx, P[1,1] = Pyy)
    if P[0, 0] > params.max_P or P[1, 1] > params.max_P:
        return True

    if state == "confirmed":
        if score < params.delete_threshold:
            return True
    else:
        # Unconfirmed (initialized or tentative)
        if score <= 0:
            return True

    return False