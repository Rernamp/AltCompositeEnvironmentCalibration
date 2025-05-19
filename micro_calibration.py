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
    residuals = np.zeros(len(rays) * 3)
    for i, ray in enumerate(rays):
        marker_index = marker_indices[i]
        if marker_index >= 0 and marker_index <= len(markers):
            marker = markers[marker_index]
            p_to_marker = marker - position
            error_residual = error_between_rays(ray, p_to_marker)
            residuals[i*3: (i+1)*3] = error_residual
    return residuals


def error_by_residuals(residuals):
    return 0.5 * np.sum(residuals ** 2)

def get_snapshot_error(snapshot, markers):
    return error_by_residuals(get_snapshot_residuals(snapshot.position, snapshot.rays, snapshot.marker_indices, markers))


def get_residuals_by_parameters(x, snapshots, markers, residuals_count, ranges):
    residuals = np.zeros(residuals_count)

    position_offsets = x[ranges.positions]

    markers_offsets = x[ranges.markers_offsets]

    new_markers = [markers[i] - markers_offsets[i] for i in range(len(markers))]

    rotate = x[ranges.rotation]

    iterator = 0

    for i, snapshot in enumerate(snapshots):
        new_position = snapshot.position - position_offsets[i]
        calc_rot = Rotation.from_euler('xyz', angles=rotate[i])
        rotated_rays = [calc_rot.apply(ray) for ray in snapshot.rays]
        rays_count = len(snapshot.rays)
        residuals[iterator:iterator + rays_count * 3] = get_snapshot_residuals(new_position, rotated_rays, snapshot.marker_indices, new_markers)
        iterator += rays_count * 3

    return residuals

def get_residuals_count(snapshots):
    residuals_count = 0
    for snapshot in snapshots:
        residuals_count += len(snapshot.rays) * 3
    return residuals_count

guess, snapshots, markers = read_dump("dataset/allAxis_1Marker_fix3/#000.json")
print(f'read {len(snapshots)} snapshots and {len(markers)} markers')

used_marker_indices = set()
for snapshot in snapshots:
    for index in snapshot.marker_indices:
        used_marker_indices.add(index)


# used_marker_indices = {index for index in [snapshot.marker_indices for snapshot in snapshots]}
print(f"Used marker indices: {used_marker_indices}")
used_markers_count = len(used_marker_indices)

snapshots_info = []

for snapshot in snapshots:
    print(f'rays: {len(snapshot.rays)} markers: {len(snapshot.marker_indices)} error: {(get_snapshot_error(snapshot, markers))}')
    snapshots_info.append(SnapshotInfo(last_result=np.array([0.0 for _ in range(3 + 3)])))

params_count = used_markers_count * 3 + len(snapshots) * 6

initial_params = np.zeros(params_count)

residuals_count = get_residuals_count(snapshots)

ranges = ParametersRanges(None, None, None)

snapshots_count = len(snapshots)

ranges.positions = [ range(i * 3,(i + 1) * 3) for i in range(snapshots_count)]
markers_index_offset = snapshots_count * 3
ranges.markers_offsets = [range(markers_index_offset + i * 3, markers_index_offset + (i + 1) * 3) for i in range(len(markers))]

rotate_index_offset = markers_index_offset + len(markers) * 3
ranges.rotation = [range(rotate_index_offset + i * 3, rotate_index_offset + (i + 1) * 3) for i in range(snapshots_count)]

iterator = 0

initial_cost = error_by_residuals(get_residuals_by_parameters(initial_params, snapshots, markers, residuals_count, ranges))

print(f"Initial cost:{initial_cost}")

max_position_offset = 0.1

lower_bound = np.ones(params_count) * -max_position_offset
lower_bound[np.array(ranges.rotation).ravel()] = np.ones(len(np.array(ranges.rotation).ravel())) * -180

upper_bound = np.ones(params_count) * max_position_offset
upper_bound[np.array(ranges.rotation).ravel()] = np.ones(len(np.array(ranges.rotation).ravel())) * 180

result = optimize.least_squares(get_residuals_by_parameters, initial_params, args=(snapshots, markers, residuals_count, ranges), bounds=(lower_bound, upper_bound))
# result = optimize.least_squares(get_residuals_by_parameters, initial_params, args=(snapshots, markers, residuals_count, ranges))

opt_cost = error_by_residuals(get_residuals_by_parameters(result.x, snapshots, markers, residuals_count, ranges))
print(f"Optimized cost:{opt_cost}")


print(f"Markers: {result.x[ranges.markers_offsets]}")

print(f"x: {result.x}")
print(f"cost: {result.cost}")
print(f"optimality: {result.optimality}")
