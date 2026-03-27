from lmfit import Parameter, Parameters, Minimizer, minimize, report_fit, fit_report
import numpy as np
import copy
import sys
from random import randrange
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *

from optimization_utils import *

def square_grid_points_centered(step, num_elements):
    side = int(np.sqrt(num_elements))
    if side * side != num_elements:
        raise ValueError(f"num_elements ({num_elements})")
    
    offset = (side - 1) * step / 2
    
    x_coords = np.arange(side) * step - offset
    y_coords = np.arange(side) * step - offset
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(num_elements)])
    
    return points

markers_from_env = square_grid_points_centered(0.1, 9)

offset = 1

positions = []

points_variants = 4

for marker in markers_from_env:
    # positions.append(marker + np.array([0, 0, offset]))
    for i in range(points_variants):
        positions.append(marker + np.array([np.random.randn(), np.random.randn(), offset]))
positions = np.array(positions)


synthetic_snapshots = []

original_markers = copy.deepcopy(markers_from_env)

for i, marker in enumerate(original_markers):
    marker += np.random.rand(marker.shape[0]) * 0.01
    # if i == 0:
    #     marker += np.random.rand(marker.shape[0]) * 0.01
        # marker[0] += 0.01

print(f"Markers: {original_markers}")
print(f"Positions: {positions}")

for i, position in enumerate(positions):
    rays = []
    marker_indices = []
    for marker_index, marker in enumerate(original_markers):
        ray = marker - position
        marker_indices.append(marker_index)
        rays.append(ray)
    synthetic_snapshots.append(
        Snapshot(position=position, rays=rays, marker_indices=marker_indices))

modified_snapshots = copy.deepcopy(synthetic_snapshots)

max_offset = 0.001

for i, snapshot in enumerate(modified_snapshots):
    # if i >= 1:
    #     snapshot.position = snapshot.position + \
    #         np.random.rand(snapshot.position.shape[0]) * max_offset
    snapshot.position = snapshot.position + \
            np.random.rand(snapshot.position.shape[0]) * max_offset
    print(f"Modify position {i} {snapshot.position}")

parameters = Parameters()

snapshot_for_optimization = modified_snapshots

for i, marker in enumerate(original_markers):
    add_point_to_parameters(parameters=parameters,
                            point=np.zeros(3), prefix=f"diff_marker_{i}", vary=True)
    
for i, snapshot in enumerate(snapshot_for_optimization):
    add_point_to_parameters(parameters=parameters,
                            point=np.zeros(3), prefix=f"diff_pos_{i}", vary=True)

for i, snapshot in enumerate(snapshot_for_optimization):
    add_point_to_parameters(parameters=parameters,
                            point=np.zeros(snapshot.position.shape), prefix=f"quat_u_{i}", vary=False)
    parameters.add(Parameter(name=f"quat_w_{i}", value=1, vary=False))


scale = 0.001
cost_func_synt = cost_func(
    parameters=parameters, snapshots=synthetic_snapshots, markers=original_markers, scale=scale)

cost_func_before = cost_func(
    parameters=parameters, snapshots=snapshot_for_optimization, markers=markers_from_env, scale=scale)

mini = Minimizer(cost_func, parameters, fcn_args=(
    snapshot_for_optimization, markers_from_env, scale))

# result = mini.minimize(method='leastsq', **
#                        {'Dfun': gradient_function, 
#                         "gtol": 1e-16,
#                         "ftol": 1e-16})
result = mini.minimize(method='leastsq', **
                       {'Dfun': gradient_function})
# result = mini.minimize(method='leastsq')
# result = mini.minimize(method='least_squares', jac=gradient_function)

with open('output.txt', 'w') as f:
    f.write(fit_report(result))
    

cost_func_after = cost_func(
    parameters=result.params, snapshots=snapshot_for_optimization, markers=markers_from_env, scale=scale)

# print(gradient_function(parameters=result.params, snapshots=snapshot_for_optimization, markers=markers))

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


total_positions_error_after = 0
total_positions_error_before = 0

for i, snapshot in enumerate(modified_snapshots):
    start_position = snapshot.position
    diff_optimized_position = extract_point_from_parameters(parameters=result.params, prefix=f"diff_pos_{i}")
    synt_positional = positions[i]
    optimized_position = start_position + diff_optimized_position
    total_positions_error_after += np.linalg.norm(optimized_position - synt_positional)
    total_positions_error_before += np.linalg.norm(start_position - synt_positional)
    
    print(f"Position {i}; True : {synt_positional}; Optimized: {optimized_position}; Start: {start_position}")

total_markers_error_after = 0
total_markers_error_before = 0

for i, synt_marker in enumerate(original_markers):
    start_marker_diff = extract_point_from_parameters(parameters=parameters, prefix=f"diff_marker_{i}")
    optimized_marker_diff = extract_point_from_parameters(parameters=result.params, prefix=f"diff_marker_{i}")
    
    modify_marker = markers_from_env[i]
    optimized_marker = optimized_marker_diff + modify_marker
    total_markers_error_after += np.linalg.norm(modify_marker + optimized_marker_diff - synt_marker)
    total_markers_error_before += np.linalg.norm(modify_marker + start_marker_diff - synt_marker)
    
    print(f"Marker {i}; True : {synt_marker}; Optimized: {optimized_marker}; Start: {modify_marker}")

print(f"Total positions error before optimization: {total_positions_error_before}")
print(f"Total positions error after optimization: {total_positions_error_after}")

print(f"Total markers error before optimization: {total_markers_error_before}")
print(f"Total markers error after optimization: {total_markers_error_after}")

print()

print(f"Synthetic sum cost func value: {np.sum(cost_func_synt)}")
print(f"Before optimization sum cost func value: {np.sum(cost_func_before)}")
print(f"After optimization sum cost func value: {np.sum(cost_func_after)}")
print(f"Func call number: {result.nfev}")

print(f"Grad func in synt data {np.sum(np.sum(gradient_function(parameters=parameters, snapshots=synthetic_snapshots, markers=markers_from_env, scale=scale)))}")