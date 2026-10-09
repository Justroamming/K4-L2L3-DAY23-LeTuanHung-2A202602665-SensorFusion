"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params

kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    mhd_sq = gamma.T @ np.linalg.inv(S) @ gamma
    return float(mhd_sq)


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    params = get_tracking_params()
    threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return mhd_sq <= threshold


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    n_tracks = len(track_list)
    n_meas = len(meas_list)
    cost_matrix = np.full((n_tracks, n_meas), np.inf, dtype=float)

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            # Check visibility first - don't call projection/Mahalanobis for invisible pairs
            if not meas.sensor.in_fov(track.x):
                continue
            # Compute Mahalanobis distance
            mhd_sq = mahalanobis_distance(track, meas)
            # Check chi-square gate
            if chi2_gate(mhd_sq, meas.sensor):
                cost_matrix[i, j] = mhd_sq

    return np.asmatrix(cost_matrix)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    # Handle empty matrix or all inf
    if association_matrix.size == 0 or np.all(np.isinf(association_matrix)):
        return (
            np.nan,
            np.nan,
            association_matrix,
            list(unassigned_tracks),
            list(unassigned_meas),
        )

    # Find minimum finite cost
    finite_mask = np.isfinite(association_matrix)
    if not np.any(finite_mask):
        return (
            np.nan,
            np.nan,
            association_matrix,
            list(unassigned_tracks),
            list(unassigned_meas),
        )

    min_idx = np.unravel_index(np.argmin(np.where(finite_mask, association_matrix, np.inf)), association_matrix.shape)
    track_idx, meas_idx = min_idx

    # Get the selected track and measurement
    selected_track = unassigned_tracks[track_idx]
    selected_meas = unassigned_meas[meas_idx]

    # Remove the selected row and column from the matrix
    new_matrix = np.delete(association_matrix, track_idx, axis=0)
    new_matrix = np.delete(new_matrix, meas_idx, axis=1)

    # Remove selected items from unassigned lists
    remaining_tracks = [t for i, t in enumerate(unassigned_tracks) if i != track_idx]
    remaining_meas = [m for i, m in enumerate(unassigned_meas) if i != meas_idx]

    return selected_track, selected_meas, np.asmatrix(new_matrix), remaining_tracks, remaining_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    # Make copies of track_list and meas_list for unassigned
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    # Build cost matrix
    cost_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)

    # Greedy association loop
    while cost_matrix.size > 0 and np.any(np.isfinite(cost_matrix)):
        track, meas, cost_matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
            cost_matrix, unassigned_tracks, unassigned_meas
        )
        # Check if no valid pair was found (returns np.nan for track/meas)
        if isinstance(track, float) and np.isnan(track):
            break

        # Update track with measurement
        filter_obj.update(track, meas)
        manager.handle_updated_track(track, sensor)

    # Always call manage_tracks, even with empty measurement lists
    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)