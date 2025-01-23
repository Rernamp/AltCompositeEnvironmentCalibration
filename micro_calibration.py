import math
import time

from EniPy import eniUtils
from dataclasses import dataclass
import numpy as np
from scipy.spatial import distance
from Utils import *
from scipy.optimize import minimize

@dataclass
class Snapshot:
    position: np.ndarray
    rays: [np.ndarray]
    marker_indices: [int]

def read_dump(path):
    raw_dump = eniUtils.readJson(path)
    snapshots = []
    for raw_snapshot in raw_dump["AltRaysSnapshots"]:
        snapshot = Snapshot(None, None, None)
        position = raw_snapshot["Position"]
        snapshot.position = np.array([position["x"], position["y"], position["z"]])
        snapshot.rays = [np.array([r["x"], r["y"], r["z"]]) for r in raw_snapshot["Rays"]]
        snapshot.marker_indices = raw_snapshot["MarkerIndices"]
        snapshots.append(snapshot)
    markers = []
    for raw_marker in raw_dump["EnvironmentMarkers"]["Markers"]:
        markers.append(np.array([raw_marker["x"], raw_marker["y"], raw_marker["z"]]))

    return snapshots, markers

def get_error(position, rays, marker_indices, markers):
    error = 0.0
    count = 0
    for i, ray in enumerate(rays):
        marker_index = marker_indices[i]
        if marker_index >= 0 and marker_index <= len(markers):
            marker = markers[marker_index]
            p_to_marker = marker - position
            error += get_angle_between_vectors(ray, p_to_marker)
            count += 1
    return error / count

def get_snapshot_error(snapshot, markers):
    return get_error(snapshot.position, snapshot.rays, snapshot.marker_indices, markers)
def f_internal(x, snapshot, markers):
    return get_error(snapshot.position + np.array(x[:3]), snapshot.rays, snapshot.marker_indices, markers)

def f_external(x, snapshots, markers):
    transformed_markers = []
    offsets = x
    for i, marker in enumerate(markers):
        transformed_markers.append(np.array(marker + offsets[:3]))
        offsets = offsets[3:]
    total_error = 0.0
    count = 0
    for snapshot in snapshots:
        begin = time.perf_counter()
        result = minimize(f_internal, np.array([0.0, 0.0, 0.0]), method='Nelder-Mead', args=(snapshot, transformed_markers))
        end = time.perf_counter()
        # print(f'completed by: {((end - begin) * 1000):1f} ms. Result: {result["success"]}')
        if result["success"]:
            x = result["x"]
            new_position = snapshot.position + x[:3]
            total_error += get_error(new_position, snapshot.rays, snapshot.marker_indices, transformed_markers)
            count += 1
    if(count < len(snapshots)):
        print(f'some internal position optimizations failed. Actual: {count} Expected {len(snapshots)}')
    return total_error / count


snapshots, markers = read_dump("micro_calibarion_dumps/10mmOffset.json")
print(f'read {len(snapshots)} snapshots and {len(markers)} markers')
initial_offsets = np.array([0.0 for _ in range(36)])
guess_offsets = initial_offsets.copy()
guess_offsets[3] = 0.01
print(f'e_initial {f_external(initial_offsets, snapshots, markers)}')
print(f'e_guess {f_external(guess_offsets, snapshots, markers)}')

for snapshot in snapshots:
    print(f'rays: {len(snapshot.rays)} markers: {len(snapshot.marker_indices)} error: {math.degrees(get_snapshot_error(snapshot, markers))}')

begin = time.perf_counter()
result = minimize(f_external, initial_offsets, method='Nelder-Mead', args=(snapshots, markers), bounds=[(-0.05, 0.05) for _ in initial_offsets])
end = time.perf_counter()
print(f'completed by: {((end - begin) * 1000):1f} ms. Result: {result}')
print(f'x: {result["x"]}')
