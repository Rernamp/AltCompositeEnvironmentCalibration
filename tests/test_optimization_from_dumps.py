from lmfit import Parameter, Parameters, Minimizer, minimize, report_fit, fit_report
import numpy as np
import copy
import sys
from random import randrange
from pathlib import Path
import rerun as rr
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *
from optimization_utils import *

RRD_PATH = str(Path(__file__).parent / "rrd_dumps" / "test_optimization_from_dumps.rrd")
Path(RRD_PATH).parent.mkdir(parents=True, exist_ok=True)
rr.init("test_optimization_from_dumps", spawn=False)
rr.save(RRD_PATH)
rr.send_blueprint(make_default_blueprint())



guess, snapshots, markers = read_dump("dataset/allAxis_1Marker_fix3/#002.json")
markers = np.array(markers)
original_markers = markers + np.array(guess)

snapshot_for_optimization = copy.deepcopy(snapshots)

for snapshot in snapshot_for_optimization:
    for i in range(len(snapshot.rays)):
        ray = original_markers[snapshot.marker_indices[i]] - snapshot.position
        ray /= np.linalg.norm(ray)
        snapshot.rays[i] = ray

parametersBuilder = ParametersBuilder(snapshots=snapshot_for_optimization, markers=markers)

# parametersBuilder.add_markers_offsets(original_markers - markers_from_env)

parameters = parametersBuilder.getResult()

scale = 0

correct_params = copy.deepcopy(parameters)
for i, marker in enumerate(markers):
    correct_diff = original_markers[i] - marker
    correct_params[f"diff_marker_{i}_x"].value = correct_diff[0]
    correct_params[f"diff_marker_{i}_y"].value = correct_diff[1]
    correct_params[f"diff_marker_{i}_z"].value = correct_diff[2]

for i in range(len(snapshot_for_optimization)):
    correct_params[f"diff_pos_{i}_x"].value = 0
    correct_params[f"diff_pos_{i}_y"].value = 0
    correct_params[f"diff_pos_{i}_z"].value = 0

cost_ground_truth = cost_func(correct_params, snapshot_for_optimization, markers, scale)
print(f"Ideal cost function value (as if all rays matched perfectly), sum of squares: {np.sum(cost_ground_truth**2):.6e}")

mini = Minimizer(cost_func, parameters, fcn_args=(
    snapshot_for_optimization, markers, scale),
    iter_cb=make_optimization_iter_cb())

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
print("=" * 80)
print("COST FUNCTION ANALYSIS")
print("=" * 80)

# Cost at initial point
cost_initial = cost_func(parameters, snapshot_for_optimization, markers, scale)
cost_initial_total = np.sum(cost_initial**2)
print(f"\n1. Initial point (from parameters):")
print(f"   Cost vector sum of squares: {cost_initial_total:.6e}")
print(f"   Cost vector mean: {np.mean(cost_initial):.6e}")
print(f"   Cost vector max: {np.max(np.abs(cost_initial)):.6e}")

# Cost at optimized point
cost_optimized = cost_func(result.params, snapshot_for_optimization, markers, scale)
cost_optimized_total = np.sum(cost_optimized**2)
print(f"\n2. Optimized point (from result.params):")
print(f"   Cost vector sum of squares: {cost_optimized_total:.6e}")
print(f"   Cost vector mean: {np.mean(cost_optimized):.6e}")
print(f"   Cost vector max: {np.max(np.abs(cost_optimized)):.6e}")

# Cost at ground truth point (gauge - correct markers & zero position diffs), computed before optimization
cost_ground_truth_total = np.sum(cost_ground_truth**2)
print(f"\n3. Ground truth point (gauge - correct markers & zero position diffs):")
print(f"   Cost vector sum of squares: {cost_ground_truth_total:.6e}")
print(f"   Cost vector mean: {np.mean(cost_ground_truth):.6e}")
print(f"   Cost vector max: {np.max(np.abs(cost_ground_truth)):.6e}")

print(f"\nImprovement (initial → optimized): {cost_initial_total / cost_optimized_total:.2f}x")
print(f"Gap to ground truth: {cost_optimized_total / cost_ground_truth_total:.2f}x")
print("=" * 80)
print()

original = np.array(original_markers)
optimized = np.array(optimized_markers)
markers_arr = np.array(markers)

estimated_markers, estimated_displacements, inliers = align_optimized_markers(markers_arr, optimized)
print(f"Detected displaced markers (indices): {np.where(~inliers)[0].tolist()}")

print(f"optimized_markers:\n{optimized}")
print(f"original_markers:\n{original}")

print("Estimated displacements per marker:")
print(estimated_displacements)

print("Estimated marker positions:")
print(estimated_markers)

print(f"Total error by markers before optimization: {np.sum(np.linalg.norm(original - markers_arr, axis=1))}")
print(f"Total error by markers after optimization:  {np.sum(np.linalg.norm(original - estimated_markers, axis=1))}")

log_points3d("world/final/original", original_markers, colors=[255, 0, 0])
log_points3d("world/final/estimated", estimated_markers, colors=[0, 255, 0])
log_points3d("world/final/markers_from_env", markers, colors=[0, 0, 255])

print(f"Rerun dump saved to {RRD_PATH}")