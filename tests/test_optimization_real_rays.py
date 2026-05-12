from lmfit import Minimizer, fit_report
import numpy as np
import copy
import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *
from optimization_utils import *


DUMP_PATH = "dataset/allAxis_1Marker_fix3/#002.json"

guess, snapshots, markers = read_dump(DUMP_PATH)
markers = np.array(markers)
original_markers = markers + np.array(guess)

snapshot_for_optimization = copy.deepcopy(snapshots)

parametersBuilder = ParametersBuilder(snapshots=snapshot_for_optimization, markers=markers)
parameters = parametersBuilder.getResult()

scale = 0

mini = Minimizer(cost_func, parameters, fcn_args=(snapshot_for_optimization, markers, scale))
result = mini.minimize(method='leastsq', **{'Dfun': gradient_function})

with open('output.txt', 'w') as f:
    f.write(fit_report(result))

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

optimized_markers = []
for i in range(len(markers)):
    optimized_marker_diff = extract_point_from_parameters(parameters=result.params, prefix=f"diff_marker_{i}")
    optimized_markers.append(markers[i] + optimized_marker_diff)
optimized_markers = np.array(optimized_markers)

print()
print("=" * 80)
print("COST FUNCTION ANALYSIS")
print("=" * 80)

cost_initial = cost_func(parameters, snapshot_for_optimization, markers, scale)
cost_initial_total = np.sum(cost_initial**2)
print(f"\n1. Initial point:")
print(f"   Cost vector sum of squares: {cost_initial_total:.6e}")
print(f"   Cost vector mean: {np.mean(cost_initial):.6e}")
print(f"   Cost vector max: {np.max(np.abs(cost_initial)):.6e}")

cost_optimized = cost_func(result.params, snapshot_for_optimization, markers, scale)
cost_optimized_total = np.sum(cost_optimized**2)
print(f"\n2. Optimized point:")
print(f"   Cost vector sum of squares: {cost_optimized_total:.6e}")
print(f"   Cost vector mean: {np.mean(cost_optimized):.6e}")
print(f"   Cost vector max: {np.max(np.abs(cost_optimized)):.6e}")

print(f"\nImprovement (initial → optimized): {cost_initial_total / cost_optimized_total:.2f}x")
print("=" * 80)
print()

estimated_markers, estimated_displacements, inliers = align_optimized_markers(markers, optimized_markers)
print(f"Detected displaced markers (indices): {np.where(~inliers)[0].tolist()}")

print(f"markers_from_env:\n{markers}")
print(f"original_markers:\n{original_markers}")
print(f"optimized_markers:\n{optimized_markers}")
print(f"estimated_markers:\n{estimated_markers}")

print("\nEstimated displacements per marker:")
print(estimated_displacements)

print(f"\nTotal error by markers before optimization: {np.sum(np.linalg.norm(original_markers - markers, axis=1)):.6f}")
print(f"Total error by markers after optimization:  {np.sum(np.linalg.norm(original_markers - estimated_markers, axis=1)):.6f}")

# 3D view
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.scatter(original_markers[:,0], original_markers[:,1], original_markers[:,2], label="Original", alpha=0.6)
ax_3d.scatter(optimized_markers[:,0], optimized_markers[:,1], optimized_markers[:,2], label="Optimized (raw)", alpha=0.6)
ax_3d.scatter(estimated_markers[:,0], estimated_markers[:,1], estimated_markers[:,2], label="Estimated (aligned)", alpha=0.6)
ax_3d.scatter(markers[:,0], markers[:,1], markers[:,2], label="markers_from_env", alpha=0.6)
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.legend()
ax_3d.set_title('3D View')

# Plane projections
fig_proj = plt.figure(figsize=(14, 10))

ax_xy = fig_proj.add_subplot(2, 2, 1)
ax_xy.scatter(original_markers[:,0], original_markers[:,1], label="Original", alpha=0.6)
ax_xy.scatter(estimated_markers[:,0], estimated_markers[:,1], label="Estimated", alpha=0.6)
ax_xy.scatter(markers[:,0], markers[:,1], label="markers_from_env", alpha=0.6)
ax_xy.set_xlabel('X')
ax_xy.set_ylabel('Y')
ax_xy.legend()
ax_xy.set_title('OXY Plane Projection')
ax_xy.grid(True)

ax_xz = fig_proj.add_subplot(2, 2, 2)
ax_xz.scatter(original_markers[:,0], original_markers[:,2], label="Original", alpha=0.6)
ax_xz.scatter(estimated_markers[:,0], estimated_markers[:,2], label="Estimated", alpha=0.6)
ax_xz.scatter(markers[:,0], markers[:,2], label="markers_from_env", alpha=0.6)
ax_xz.set_xlabel('X')
ax_xz.set_ylabel('Z')
ax_xz.legend()
ax_xz.set_title('OXZ Plane Projection')
ax_xz.grid(True)

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
