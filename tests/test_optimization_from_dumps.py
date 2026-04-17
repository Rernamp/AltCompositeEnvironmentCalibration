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


max_offset = 0.005

guess, snapshots, markers = read_dump("dataset/allAxis_1Marker_fix3/#002.json")
markers = np.array(markers)
original_markers = markers + np.array(guess)

snapshot_for_optimization = copy.deepcopy(snapshots)

for snapshot in snapshot_for_optimization:
    for i, ray in enumerate(snapshot.rays):
        ray = markers[snapshot.marker_indices[i]] - snapshot.position
        ray /= np.linalg.norm(ray)        

parametersBuilder = ParametersBuilder(snapshots=snapshot_for_optimization, markers=markers)

# parametersBuilder.add_markers_offsets(original_markers - markers_from_env)

parameters = parametersBuilder.getResult()

scale = 1

mini = Minimizer(cost_func, parameters, fcn_args=(
    snapshot_for_optimization, markers, scale))

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
    
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


total_positions_error_after = 0
total_positions_error_before = 0

total_markers_error_after = 0
total_markers_error_before = 0

optimized_markers = []

for i, synt_marker in enumerate(original_markers):
    start_marker_diff = extract_point_from_parameters(parameters=parameters, prefix=f"diff_marker_{i}")
    optimized_marker_diff = extract_point_from_parameters(parameters=result.params, prefix=f"diff_marker_{i}")
    
    modify_marker = markers[i]
    optimized_marker = optimized_marker_diff + modify_marker
    total_markers_error_after += np.linalg.norm(modify_marker + optimized_marker_diff - synt_marker)
    total_markers_error_before += np.linalg.norm(modify_marker + start_marker_diff - synt_marker)
    optimized_markers.append(np.array(optimized_marker))
    print(f"Marker {i}; True : {synt_marker}; Optimized: {optimized_marker}; Start: {modify_marker}")
optimized_markers = np.array(optimized_markers)

print(f"Total markers error before optimization: {total_markers_error_before}")
print(f"Total markers error after optimization: {total_markers_error_after}")

print()

print(f"Func call number: {result.nfev}")

original = np.array(original_markers)
optimized = np.array(optimized_markers)


M_aligned = recover_target(optimized_markers, markers)

print(f"optimized_markers:\n{optimized}")
print(f"original_markers:\n{original}")

print("Aligned markers (without scaling):")
print(M_aligned)

print(f"Total error by markers before optimization: {np.sum(np.linalg.norm(original - np.array(markers), axis=1))}")
print(f"Total error by markers after optimization: {np.sum(np.linalg.norm(original - M_aligned, axis=1))}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(original_markers[:,0], original_markers[:,1], original_markers[:,2], label="Original")
ax.scatter(optimized_markers[:,0], optimized_markers[:,1], optimized_markers[:,2], label="Optimized")
ax.scatter(M_aligned[:,0], M_aligned[:,1], M_aligned[:,2], label="Aligned")
ax.scatter(markers[:,0], markers[:,1], markers[:,2], label="markers_from_env")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()