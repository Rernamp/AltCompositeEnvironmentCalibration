from lmfit import Parameter, Parameters, Minimizer, minimize, report_fit, fit_report
import numpy as np
import copy
import sys
from random import randrange
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *
from scipy.linalg  import orthogonal_procrustes
from scipy.spatial import procrustes

from optimization_utils import *


markers_from_env = square_grid_points_centered(0.5, 9)


points_variants = 4

rng = np.random.default_rng(123)

positions = generate_offset_variants(markers_from_env, np.array([0, 0, 1]), points_variants, 0.01, rng)
original_markers = generate_offset_variants(markers_from_env, np.array([0, 0, 0]), 1, 0.005, rng)

print(f"Markers: {original_markers}")
print(f"Positions: {positions}")

synthetic_snapshots = generate_snapshots(positions=positions, markers=original_markers)

max_offset = 0.005

modify_positions = generate_offset_variants(positions, np.array([0, 0, 0]), 1, max_offset, rng)

modified_snapshots = generate_snapshots(modify_positions, original_markers)

snapshot_for_optimization = modified_snapshots
parametersBuilder = ParametersBuilder(snapshots=snapshot_for_optimization, markers=markers_from_env)

# parametersBuilder.add_markers_offsets(original_markers - markers_from_env)

parameters = parametersBuilder.getResult()

scale = 1
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

optimized_markers = []

for i, synt_marker in enumerate(original_markers):
    start_marker_diff = extract_point_from_parameters(parameters=parameters, prefix=f"diff_marker_{i}")
    optimized_marker_diff = extract_point_from_parameters(parameters=result.params, prefix=f"diff_marker_{i}")
    
    modify_marker = markers_from_env[i]
    optimized_marker = optimized_marker_diff + modify_marker
    total_markers_error_after += np.linalg.norm(modify_marker + optimized_marker_diff - synt_marker)
    total_markers_error_before += np.linalg.norm(modify_marker + start_marker_diff - synt_marker)
    optimized_markers.append(np.array(optimized_marker))
    print(f"Marker {i}; True : {synt_marker}; Optimized: {optimized_marker}; Start: {modify_marker}")
optimized_markers = np.array(optimized_markers)
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



original = np.array(original_markers)
optimized = np.array(optimized_markers)


M_aligned = recover_target(optimized_markers, original_markers)

print(f"optimized_markers:\n{optimized}")
print(f"original_markers:\n{original}")

print("Aligned markers (without scaling):")
print(M_aligned)

print(f"Total error by markers before optimization: {np.sum(np.linalg.norm(original - np.array(markers_from_env), axis=1))}")
print(f"Total error by markers after optimization: {np.sum(np.linalg.norm(original - M_aligned, axis=1))}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(original_markers[:,0], original_markers[:,1], original_markers[:,2], label="Original")
ax.scatter(optimized_markers[:,0], optimized_markers[:,1], optimized_markers[:,2], label="Optimized")
ax.scatter(M_aligned[:,0], M_aligned[:,1], M_aligned[:,2], label="Aligned")
ax.scatter(markers_from_env[:,0], markers_from_env[:,1], markers_from_env[:,2], label="markers_from_env")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()