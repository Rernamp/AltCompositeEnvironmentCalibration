import math
import time
import datetime
import numpy as np
from scipy.spatial import distance
from Utils import *
from scipy.spatial.transform import Rotation
from lmfit import Parameters, minimize, report_fit

def get_snapshot_residuals(position, rays, marker_indices, markers):
    residuals = np.zeros((len(rays), 3))
    for i, ray in enumerate(rays):
        marker_index = marker_indices[i]
        if 0 <= marker_index < len(markers):
            marker = markers[marker_index]
            p_to_marker = marker - position
            error_residual = error_between_rays(ray, p_to_marker)
            residuals[i] = error_residual
    return residuals


def error_by_residuals(residuals):
    return 0.5 * np.sum(residuals ** 2)


def get_snapshot_error(snapshot, markers):
    return error_by_residuals(
        get_snapshot_residuals(snapshot.position, snapshot.rays, snapshot.marker_indices, markers))


def create_lmfit_parameters(snapshots, markers, initial_guess=None, max_position_offset=0.02):
    params = Parameters()

    # Add position offsets for each snapshot
    for i in range(len(snapshots)):
        for j in range(3):
            params.add(f'pos_{i}_{j}', value=0)

    # Add marker offsets
    for i in range(len(markers)):
        for j in range(3):
            val = initial_guess[i][j] if initial_guess is not None else 0
            params.add(f'marker_{i}_{j}', value=val)

    # Add rotation parameters for each snapshot
    for i in range(len(snapshots)):
        for j in range(3):
            params.add(f'rot_{i}_{j}', value=0, min=-180, max=180)

    return params


def snapshots_residuals(params, snapshots, markers):
    residuals = []

    # Extract parameters
    pos_offsets = []
    rot_params = []
    marker_offsets = []

    snap_count = len(snapshots)
    marker_count = len(markers)

    for i in range(snap_count):
        pos_offsets.append([params[f'pos_{i}_{j}'].value for j in range(3)])
        rot_params.append([params[f'rot_{i}_{j}'].value for j in range(3)])

    for i in range(marker_count):
        marker_offsets.append([params[f'marker_{i}_{j}'].value for j in range(3)])

    # Calculate residuals
    new_markers = [markers[i] + np.array(marker_offsets[i]) for i in range(marker_count)]

    for i, snapshot in enumerate(snapshots):
        new_position = snapshot.position + np.array(pos_offsets[i])
        calc_rot = Rotation.from_euler('xyz', angles=rot_params[i])
        rotated_rays = [calc_rot.apply(ray) for ray in snapshot.rays]

        for j, ray in enumerate(rotated_rays):
            marker_index = snapshot.marker_indices[j]
            if 0 <= marker_index < marker_count:
                marker = new_markers[marker_index]
                p_to_marker = marker - new_position
                error_residual = error_between_rays(ray, p_to_marker)
                residuals.append(error_residual)

    return np.array(residuals)


def callback_lmfit(params, iter, resid, *args, **kws):
    if iter % 1000 == 0:
        current_error = 0.5 * np.sum(resid ** 2)
        print(f'{datetime.datetime.now()} Iteration {iter}, error: {current_error}')

def optimize_with_lmfit(snapshots, markers, initial_guess=None, grad_scale=None):
    params = create_lmfit_parameters(snapshots, markers, initial_guess)

    result = minimize(
        snapshots_residuals,
        params,
        args=(snapshots, markers),
        method='leastsq',
        iter_cb=callback_lmfit,
        col_deriv=1
    )

    return result

# Main execution
if __name__ == "__main__":
    guess, snapshots, markers = read_dump("dataset/allAxis_1Marker_fix3/#000.json")
    print(f'read {len(snapshots)} snapshots and {len(markers)} markers')

    used_marker_indices = set()
    for snapshot in snapshots:
        for index in snapshot.marker_indices:
            used_marker_indices.add(index)

    print(f"Used marker indices: {used_marker_indices}")

    # Calculate initial error
    initial_error = sum(get_snapshot_error(snapshot, markers) for snapshot in snapshots)
    print(f"Initial total error: {initial_error}")

    # Run optimization
    begin = time.perf_counter()
    result = optimize_with_lmfit(snapshots, markers, initial_guess=guess)
    end = time.perf_counter()

    print(f'Optimization completed in {((end - begin) * 1000):.1f} ms')
    # report_fit(result)

    # Extract and print results
    marker_offsets = []
    for i in range(len(markers)):
        marker_offsets.append([result.params[f'marker_{i}_{j}'].value for j in range(3)])

    print("\nOptimized marker offsets:")
    for i, offset in enumerate(marker_offsets):
        print(f"Marker {i}: {offset}")

    # Calculate final error
    final_residuals = snapshots_residuals(result.params, snapshots, markers)
    final_error = 0.5 * np.sum(final_residuals ** 2)
    print(f"\nFinal error: {final_error}")