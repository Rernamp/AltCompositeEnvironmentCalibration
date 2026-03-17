from Utils import *
from lmfit import Parameter, Parameters, Minimizer, minimize, report_fit
import numpy as np
import copy
import sys
from random import randrange
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


markers = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])

z_offset = 1

positions = np.array([(marker + np.array([0, 0, z_offset]))
                     for marker in markers])

print(f"Markers: {markers}")
print(f"Positions: {positions}")

synthetic_snapshots = []

for i, position in enumerate(positions):
    position = position
    rays = []
    marker_indices = []
    for marker_index, marker in enumerate(markers):
        ray = marker - position
        marker_indices.append(marker_index)
        rays.append(ray)
    synthetic_snapshots.append(
        Snapshot(position=position, rays=rays, marker_indices=marker_indices))


modified_snapshots = copy.deepcopy(synthetic_snapshots)


max_offset = 0.1

for i, snapshot in enumerate(modified_snapshots):
    snapshot.position = snapshot.position + \
        np.random.rand(snapshot.position.shape[0]) * max_offset
    print(f"Modify position {i} {snapshot.position}")

parameters = Parameters()

for i, snapshot in enumerate(modified_snapshots):
    add_point_to_parameters(parameters=parameters,
                            point=snapshot.position, prefix=f"pos_{i}")

print(parameters)


def cost_func(parameters, snapshots, markers):
    return 0
    
mini = Minimizer(cost_func, parameters, fcn_args=(modified_snapshots, markers))
result = mini.minimize(method='BFGS')
print(report_fit(result))
    
    