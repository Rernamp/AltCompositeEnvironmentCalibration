from lmfit import Minimizer, fit_report
import numpy as np
import copy
import sys
from pathlib import Path
import rerun as rr
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *
from optimization_utils import *

RRD_PATH = str(Path(__file__).parent / "rrd_dumps" / "test_optimization_real_rays.rrd")
Path(RRD_PATH).parent.mkdir(parents=True, exist_ok=True)
rr.init("test_optimization_real_rays", spawn=False)
rr.save(RRD_PATH)
rr.send_blueprint(make_default_blueprint())


DUMP_PATH = "dataset/allAxis_1Marker_fix3/#002.json"

guess, snapshots, markers = read_dump(DUMP_PATH)
markers = np.array(markers)
original_markers = markers + np.array(guess)

snapshot_for_optimization = copy.deepcopy(snapshots)

parametersBuilder = ParametersBuilder(snapshots=snapshot_for_optimization, markers=markers)
parameters = parametersBuilder.getResult()

scale = 0

# Ideal cost: fix markers at their true (guessed) positions and re-fit only the
# camera positions to the real (measured) rays. This is the best achievable cost
# given real ray noise, i.e. "as if all rays matched" with correct marker positions.
ideal_params = copy.deepcopy(parameters)
for i, marker in enumerate(markers):
    correct_diff = original_markers[i] - marker
    ideal_params[f"diff_marker_{i}_x"].set(value=correct_diff[0], vary=False)
    ideal_params[f"diff_marker_{i}_y"].set(value=correct_diff[1], vary=False)
    ideal_params[f"diff_marker_{i}_z"].set(value=correct_diff[2], vary=False)

ideal_mini = Minimizer(cost_func, ideal_params, fcn_args=(snapshot_for_optimization, markers, scale),
    iter_cb=make_optimization_iter_cb(timeline="ideal_iteration", entity_prefix="ideal/"))
ideal_result = ideal_mini.minimize(method='leastsq', Dfun=gradient_function)
cost_ideal = cost_func(ideal_result.params, snapshot_for_optimization, markers, scale)
print(f"Ideal cost function value (markers fixed at true positions, positions re-fit to real rays), sum of squares: {np.sum(cost_ideal**2):.6e}")

mini = Minimizer(cost_func, parameters, fcn_args=(snapshot_for_optimization, markers, scale),
    iter_cb=make_optimization_iter_cb())
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

log_points3d("world/final/original", original_markers, colors=[255, 0, 0])
log_points3d("world/final/optimized_raw", optimized_markers, colors=[255, 255, 0])
log_points3d("world/final/estimated", estimated_markers, colors=[0, 255, 0])
log_points3d("world/final/markers_from_env", markers, colors=[0, 0, 255])

print(f"Rerun dump saved to {RRD_PATH}")
