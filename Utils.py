import numpy as np
from dataclasses import dataclass
from EniPy import eniUtils
from scipy.spatial.transform import Rotation

def get_metrics(collection):
    return {'min': min(collection), 'max': max(collection), 'average': float(np.average(collection))}

def print_table(name, data):
    print(f'{name}')
    for k in sorted(data.keys()):
        str = ''
        for v in sorted(data[k].values()):
            str += f'{v:.2f}\t'
        print(str)

def get_angle_between_quaternions(q0, q1):
    return (q0 * q1.inv()).magnitude()

def get_angle_between_vectors(v0, v1):
    v0_u = v0 / np.linalg.norm(v0)
    v1_u = v1 / np.linalg.norm(v1)
    return np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))

def error_between_rays(ray, point_to_marker):
    # return get_angle_between_vectors(ray, point_to_marker)
    error = ray - point_to_marker / np.linalg.norm(point_to_marker)
    return error

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


def unpackParameters(params, snapshots, markers):

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

    return pos_offsets, rot_params, marker_offsets