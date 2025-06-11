import math
import time
import datetime
from EniPy import eniUtils
from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance
from Utils import *
from scipy.optimize import minimize
from scipy import optimize
from scipy.spatial.transform import Rotation
from lmfit import Parameters, minimize, report_fit


@dataclass
class Snapshot:
    position: np.ndarray
    rays: [np.ndarray]
    marker_indices: [int]


@dataclass
class Snapshots:
    snapsots: [Snapshot]
    markers_indices_to_params_indices: dict[int, int]


@dataclass
class ParametersRanges:
    positions: [np.ndarray]
    rotation: [np.ndarray]
    markers_offsets: [np.ndarray]


@dataclass
class SnapshotInfo:
    last_result: np.ndarray


def read_dump(path):
    file = eniUtils.readJson(path)
    raw_dump = file["dump"]
    snapshots = []
    for raw_snapshot in raw_dump["snapshots"]:
        snapshot = Snapshot(None, None, None)
        position = raw_snapshot["cameraPosition"]
        snapshot.position = np.array(position)
        snapshot.rays = [np.array(r) for r in raw_snapshot["rays"]]
        snapshot.marker_indices = raw_snapshot["markerIndices"]
        snapshots.append(snapshot)
    markers = []
    for raw_marker in raw_dump["markersPositions"]:
        markers.append(np.array(raw_marker))

    return file["offsets"], snapshots, markers


def get_snapshot_residuals(position, rays, marker_indices, markers):
    residuals = np.zeros(len(rays))
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


def create_lmfit_parameters(snapshots, markers, initial_guess=None, max_position_offset=1):
    params = Parameters()

    # Add position offsets for each snapshot
    for i in range(len(snapshots)):
        for j in range(3):
            params.add(f'pos_{i}_{j}', value=0, min=-max_position_offset, max=max_position_offset)

    # Add marker offsets
    for i in range(len(markers)):
        for j in range(3):
            val = initial_guess[i][j] if initial_guess is not None else 0
            params.add(f'marker_{i}_{j}', value=val, min=-max_position_offset, max=max_position_offset)

    # Add rotation parameters for each snapshot
    for i in range(len(snapshots)):
        for j in range(3):
            params.add(f'rot_{i}_{j}', value=0, min=-180, max=180)

    return params


def residuals_lmfit(params, snapshots, markers):
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
        # You can add more detailed reporting here if needed


def compute_analytical_jacobian(params, snapshots, markers, grad_scale=None):
    """Аналитический расчет якобиана с поддержкой масштабирования градиентов"""
    # Получаем имена параметров в порядке их объявления
    param_names = list(params.keys())
    n_params = len(param_names)
    n_residuals = sum(len(s.rays) for s in snapshots)

    # Инициализация якобиана
    jac = np.zeros((n_residuals, n_params))

    # Если grad_scale не задан, используем единичный масштаб
    if grad_scale is None:
        grad_scale = np.ones(n_residuals)
    elif len(grad_scale) != n_residuals:
        raise ValueError(f"grad_scale must have length {n_residuals}, got {len(grad_scale)}")

    residual_idx = 0

    for snap_idx, snapshot in enumerate(snapshots):
        # Текущие параметры
        pos_offset = np.array([params[f'pos_{snap_idx}_{j}'].value for j in range(3)])
        rot = np.array([params[f'rot_{snap_idx}_{j}'].value for j in range(3)])
        current_pos = snapshot.position + pos_offset
        rotation = Rotation.from_euler('xyz', rot)

        for ray_idx, ray in enumerate(snapshot.rays):
            marker_idx = snapshot.marker_indices[ray_idx]
            if 0 <= marker_idx < len(markers):
                marker_offset = np.array([params[f'marker_{marker_idx}_{j}'].value for j in range(3)])
                marker = markers[marker_idx] + marker_offset
                point_to_marker = marker - current_pos
                norm = np.linalg.norm(point_to_marker)
                normalized = point_to_marker / norm
                rotated_ray = rotation.apply(ray)
                error = rotated_ray - normalized

                # Нормализованная ошибка
                error_norm = np.linalg.norm(error)
                if error_norm == 0:
                    d_norm_error = np.zeros(3)
                else:
                    d_norm_error = error / error_norm

                # Масштаб для текущей невязки
                current_scale = grad_scale[residual_idx + ray_idx]

                # Производные по позиции камеры
                for pos_param in range(3):
                    param_name = f'pos_{snap_idx}_{pos_param}'
                    param_idx = param_names.index(param_name)

                    term1 = point_to_marker * point_to_marker[pos_param] / norm ** 3
                    term2 = np.zeros(3)
                    term2[pos_param] = 1.0
                    d_norm_d_pos = term1 - term2 / norm

                    jac[residual_idx + ray_idx, param_idx] = -current_scale * np.dot(d_norm_error, d_norm_d_pos)

                # Производные по смещениям маркеров
                for marker_param in range(3):
                    param_name = f'marker_{marker_idx}_{marker_param}'
                    param_idx = param_names.index(param_name)

                    term1 = point_to_marker * point_to_marker[marker_param] / norm ** 3
                    term2 = np.zeros(3)
                    term2[marker_param] = 1.0
                    d_norm_d_marker = (term2 / norm - term1)

                    jac[residual_idx + ray_idx, param_idx] = current_scale * np.dot(d_norm_error, d_norm_d_marker)

                # Производные по углам вращения
                for rot_param in range(3):
                    param_name = f'rot_{snap_idx}_{rot_param}'
                    param_idx = param_names.index(param_name)

                    delta = 1e-6
                    rot_plus = rot.copy()
                    rot_plus[rot_param] += delta
                    rot_minus = rot.copy()
                    rot_minus[rot_param] -= delta

                    R_plus = Rotation.from_euler('xyz', rot_plus)
                    R_minus = Rotation.from_euler('xyz', rot_minus)

                    d_rotated_ray = (R_plus.apply(ray) - R_minus.apply(ray)) / (2 * delta)
                    jac[residual_idx + ray_idx, param_idx] = current_scale * np.dot(d_norm_error, d_rotated_ray)

        residual_idx += len(snapshot.rays)

    return jac



def callback_lmfit(params, iter, resid, *args, **kws):
    if iter % 1000 == 0:
        current_error = 0.5 * np.sum(resid**2)
        print(f'{datetime.datetime.now()} Iteration {iter}, error: {current_error}')
        # You can add more detailed reporting here if needed


def optimize_with_lmfit(snapshots, markers, initial_guess=None, grad_scale=None):
    params = create_lmfit_parameters(snapshots, markers, initial_guess)

    # Обертка для передачи grad_scale в якобиан
    def jac_wrapper(params, snapshots, markers):
        return compute_analytical_jacobian(params, snapshots, markers, grad_scale)

    result = minimize(
        residuals_lmfit,
        params,
        args=(snapshots, markers),
        method='leastsq',
        # Dfun=jac_wrapper,
        iter_cb=callback_lmfit,
        col_deriv=1
    )

    return result

# Main execution
if __name__ == "__main__":
    guess, snapshots, markers = read_dump("dataset/allAxis_1Marker_fix3/#001.json")
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
    report_fit(result)

    # Extract and print results
    marker_offsets = []
    for i in range(len(markers)):
        marker_offsets.append([result.params[f'marker_{i}_{j}'].value for j in range(3)])

    print("\nOptimized marker offsets:")
    for i, offset in enumerate(marker_offsets):
        print(f"Marker {i}: {offset}")

    # Calculate final error
    final_residuals = residuals_lmfit(result.params, snapshots, markers)
    final_error = 0.5 * np.sum(final_residuals ** 2)
    print(f"\nFinal error: {final_error}")