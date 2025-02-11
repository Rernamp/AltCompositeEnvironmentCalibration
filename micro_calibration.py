import math
import time

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
class SnapshotInfo:
    last_result: np.ndarray

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
    errors = []
    for i, ray in enumerate(rays):
        marker_index = marker_indices[i]
        if marker_index >= 0 and marker_index <= len(markers):
            marker = markers[marker_index]
            p_to_marker = marker - position
            errors.append(get_angle_between_vectors(ray, p_to_marker))
    return np.average(errors)

def get_snapshot_error(snapshot, markers):
    return get_error(snapshot.position, snapshot.rays, snapshot.marker_indices, markers)
def f_internal(x, snapshot, markers):
    rotation = Rotation.from_euler('xyz', angles=x[3:6])
    return get_error(snapshot.position + np.array(x[:3]), [rotation.apply(ray) for ray in snapshot.rays], snapshot.marker_indices, markers)

def f_external(x, snapshots, markers, snapshots_info = None):
    transformed_markers = [np.array(markers[i] + x[i * 3 : (i + 1 ) * 3]) for i in range(len(markers))]
    errors = []
    for i, snapshot in enumerate(snapshots):
        begin = time.perf_counter()
        if snapshots_info != None:
            initial_x = snapshots_info[i].last_result
        else:
            initial_x = np.array([0.0 for _ in range(3 + 3)])
        result = minimize(f_internal, initial_x, method='Nelder-Mead', args=(snapshot, transformed_markers))
        end = time.perf_counter()
        # print(f'completed by: {((end - begin) * 1000):1f} ms. Result: {result["success"]}')
        if result["success"]:
            if snapshots_info != None:
                snapshots_info[i].last_result = result["x"]
            calc_offset = np.array(result["x"][:3])
            calc_rot = Rotation.from_euler('xyz', angles=result["x"][3:6])
            new_position = snapshot.position + calc_offset
            e = get_error(new_position, [calc_rot.apply(ray) for ray in snapshot.rays], snapshot.marker_indices, transformed_markers)
            # e += (np.linalg.norm(calc_offset) + calc_rot.magnitude()) / 1000
            errors.append(e)
    if(len(errors) < len(snapshots)):
        print(f'some internal position optimizations failed. Actual: {len(errors)} Expected {len(snapshots)}')
    return np.average(errors)

def on_new_minimum(x, f, context):
    print(f'new f = {f}')
    for i in range(len(x) // 3):
        m_offset = x[i * 3: (i + 1) * 3]
        print(f'{i:3}: {m_offset[0] * 1000:5.2f} {m_offset[1] * 1000:5.2f} {m_offset[2] * 1000:5.2f}')

def on_new_minimum2(xk):
    print(f'new minimum xk = {xk}')

snapshots, markers = read_dump("micro_calibarion_dumps/10mmOffset.json")
print(f'read {len(snapshots)} snapshots and {len(markers)} markers')
initial_offsets = np.array([0.0 for _ in range(len(markers) * 3)])
guess_offsets = initial_offsets.copy()
guess_offsets[3] = 0.01

print(f'e_initial {f_external(initial_offsets, snapshots, markers)}')
print(f'e_guess {f_external(guess_offsets, snapshots, markers)}')

snapshots_info = []

for snapshot in snapshots:
    print(f'rays: {len(snapshot.rays)} markers: {len(snapshot.marker_indices)} error: {math.degrees(get_snapshot_error(snapshot, markers))}')
    snapshots_info.append(SnapshotInfo(last_result=np.array([0.0 for _ in range(3 + 3)])))

begin = time.perf_counter()
# result = optimize.shgo(f_external, bounds=[(-0.05, 0.05) for _ in initial_offsets], args=(snapshots, markers, snapshots_info))
#result = optimize.dual_annealing(f_external, bounds=[(-0.02, 0.02) for _ in initial_offsets], args=(snapshots, markers), x0 = initial_offsets, callback=on_new_minimum)
result_global = optimize.direct(f_external, bounds=[(-0.05, 0.05) for _ in initial_offsets], args=(snapshots, markers, snapshots_info), callback=on_new_minimum2)
end = time.perf_counter()
print(f'completed global by: {((end - begin) * 1000):1f} ms. Result: {result_global}')
print(f'Offsets in mm:')
for i in range(len(markers)):
    m_offset = result_global["x"][i * 3 : (i + 1) * 3]
    print(f'{i:3}: {m_offset[0] * 1000:5.2f} {m_offset[1] * 1000:5.2f} {m_offset[2] * 1000:5.2f}')

begin = time.perf_counter()
result = minimize(f_external, result_global["x"], method='Nelder-Mead', args=(snapshots, markers, snapshots_info), bounds=[(-0.05, 0.05) for _ in initial_offsets])
end = time.perf_counter()
print(f'completed local by: {((end - begin) * 1000):1f} ms. Result: {result}')
print(f'Offsets in mm:')
for i in range(len(markers)):
    m_offset = result["x"][i * 3 : (i + 1) * 3]
    print(f'{i:3}: {m_offset[0] * 1000:5.2f} {m_offset[1] * 1000:5.2f} {m_offset[2] * 1000:5.2f}')
