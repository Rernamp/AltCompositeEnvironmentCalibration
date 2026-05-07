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



guess, snapshots, markers = read_dump("dataset/allAxis_1Marker_fix3/#001.json")
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

# Cost at ground truth point (gauge - correct positions)
# Create parameters with correct diff_markers
correct_params = copy.deepcopy(parameters)
for i, marker in enumerate(markers):
    correct_diff = original_markers[i] - marker
    correct_params[f"diff_marker_{i}_x"].value = correct_diff[0]
    correct_params[f"diff_marker_{i}_y"].value = correct_diff[1]
    correct_params[f"diff_marker_{i}_z"].value = correct_diff[2]

# All position diffs should be zero at ground truth
for i in range(len(snapshot_for_optimization)):
    correct_params[f"diff_pos_{i}_x"].value = 0
    correct_params[f"diff_pos_{i}_y"].value = 0
    correct_params[f"diff_pos_{i}_z"].value = 0

cost_ground_truth = cost_func(correct_params, snapshot_for_optimization, markers, scale)
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

# Fit env markers into the optimized frame, then un-apply the gauge transform
# to recover per-marker displacements in the env frame.
# optimized ≈ scale * R * env + T  (scale+translation gauge drift)
# residuals in optimized frame = optimized - fitted_env ≈ scale * true_displacement
# un-apply scale and rotation → displacements in env frame
scale_gauge, R_gauge, t_gauge = similarity_transform_svd(markers_arr, optimized)
fitted_env = scale_gauge * (markers_arr @ R_gauge.T) + t_gauge
rough_displacements = (1 / scale_gauge) * ((optimized - fitted_env) @ R_gauge)

# Pass 2: refit using only inliers so displaced markers don't skew gauge removal.
# Tukey fences on residual norms — robust to up to ~25% displaced markers.
residual_norms = np.linalg.norm(rough_displacements, axis=1)
q1, q3 = np.percentile(residual_norms, [25, 75])
inliers = residual_norms <= q3 + 1.5 * (q3 - q1)

scale_gauge, R_gauge, t_gauge = similarity_transform_svd(markers_arr[inliers], optimized[inliers])
fitted_env_all = scale_gauge * (markers_arr @ R_gauge.T) + t_gauge
estimated_displacements = (1 / scale_gauge) * ((optimized - fitted_env_all) @ R_gauge)
estimated_markers = markers_arr + estimated_displacements

print(f"Detected displaced markers (indices): {np.where(~inliers)[0].tolist()}")
estimated_markers = markers_arr + estimated_displacements

print(f"optimized_markers:\n{optimized}")
print(f"original_markers:\n{original}")

print("Estimated displacements per marker:")
print(estimated_displacements)

print("Estimated marker positions:")
print(estimated_markers)

print(f"Total error by markers before optimization: {np.sum(np.linalg.norm(original - markers_arr, axis=1))}")
print(f"Total error by markers after optimization:  {np.sum(np.linalg.norm(original - estimated_markers, axis=1))}")

# 3D view on separate figure
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.scatter(original_markers[:,0], original_markers[:,1], original_markers[:,2], label="Original", alpha=0.6)
ax_3d.scatter(estimated_markers[:,0], estimated_markers[:,1], estimated_markers[:,2], label="Estimated", alpha=0.6)
ax_3d.scatter(markers[:,0], markers[:,1], markers[:,2], label="markers_from_env", alpha=0.6)
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.legend()
ax_3d.set_title('3D View')

# Plane projections on separate figure
fig_proj = plt.figure(figsize=(14, 10))

# OXY plane projection (Z=0)
ax_xy = fig_proj.add_subplot(2, 2, 1)
ax_xy.scatter(original_markers[:,0], original_markers[:,1], label="Original", alpha=0.6)
ax_xy.scatter(estimated_markers[:,0], estimated_markers[:,1], label="Estimated", alpha=0.6)
ax_xy.scatter(markers[:,0], markers[:,1], label="markers_from_env", alpha=0.6)
ax_xy.set_xlabel('X')
ax_xy.set_ylabel('Y')
ax_xy.legend()
ax_xy.set_title('OXY Plane Projection')
ax_xy.grid(True)

# OXZ plane projection (Y=0)
ax_xz = fig_proj.add_subplot(2, 2, 2)
ax_xz.scatter(original_markers[:,0], original_markers[:,2], label="Original", alpha=0.6)
ax_xz.scatter(estimated_markers[:,0], estimated_markers[:,2], label="Estimated", alpha=0.6)
ax_xz.scatter(markers[:,0], markers[:,2], label="markers_from_env", alpha=0.6)
ax_xz.set_xlabel('X')
ax_xz.set_ylabel('Z')
ax_xz.legend()
ax_xz.set_title('OXZ Plane Projection')
ax_xz.grid(True)

# OYZ plane projection (X=0)
ax_yz = fig_proj.add_subplot(2, 2, 3)
ax_yz.scatter(original_markers[:,1], original_markers[:,2], label="Original", alpha=0.6)
ax_yz.scatter(estimated_markers[:,1], estimated_markers[:,2], label="Estimated", alpha=0.6)
ax_yz.scatter(markers[:,1], markers[:,2], label="markers_from_env", alpha=0.6)
ax_yz.set_xlabel('Y')
ax_yz.set_ylabel('Z')
ax_yz.legend()
ax_yz.set_title('OYZ Plane Projection')
ax_yz.grid(True)

fig_proj.tight_layout()
fig_3d.tight_layout()
plt.show()