from lmfit import Minimizer
import numpy as np
import copy
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *
from optimization_utils import *


def run_optimization(dump_path: str):
    guess, snapshots, markers = read_dump(dump_path)
    markers = np.array(markers)
    original_markers = markers + np.array(guess)

    snapshot_for_optimization = copy.deepcopy(snapshots)

    for snapshot in snapshot_for_optimization:
        for i in range(len(snapshot.rays)):
            ray = original_markers[snapshot.marker_indices[i]] - snapshot.position
            ray /= np.linalg.norm(ray)
            snapshot.rays[i] = ray

    parametersBuilder = ParametersBuilder(snapshots=snapshot_for_optimization, markers=markers)
    parameters = parametersBuilder.getResult()

    scale = 0
    mini = Minimizer(cost_func, parameters, fcn_args=(snapshot_for_optimization, markers, scale))
    result = mini.minimize(method='leastsq', **{'Dfun': gradient_function})

    optimized_markers = []
    for i, synt_marker in enumerate(original_markers):
        optimized_marker_diff = extract_point_from_parameters(parameters=result.params, prefix=f"diff_marker_{i}")
        optimized_markers.append(markers[i] + optimized_marker_diff)
    optimized_markers = np.array(optimized_markers)

    estimated_markers, _, _ = align_optimized_markers(markers, optimized_markers)

    error_before = np.sum(np.linalg.norm(original_markers - markers, axis=1))
    error_after = np.sum(np.linalg.norm(original_markers - estimated_markers, axis=1))
    return error_before, error_after


def main():
    parser = argparse.ArgumentParser(description="Batch optimization over dump files in a folder")
    parser.add_argument("folder", help="Path to folder with .json dump files")
    args = parser.parse_args()

    folder = Path(args.folder)
    dump_files = sorted(folder.glob("*.json"))

    if not dump_files:
        print(f"No .json files found in {folder}")
        return

    total_before = 0.0
    total_after = 0.0
    results = []

    for dump_path in dump_files:
        try:
            error_before, error_after = run_optimization(str(dump_path))
            results.append((dump_path.name, error_before, error_after))
            total_before += error_before
            total_after += error_after
            print(f"{dump_path.name}: before={error_before:.6f}  after={error_after:.6f}")
        except Exception as e:
            print(f"{dump_path.name}: ERROR - {e}")

    print()
    print("=" * 60)
    print(f"Files processed : {len(results)}")
    print(f"Total error before: {total_before:.6f}")
    print(f"Total error after : {total_after:.6f}")
    if total_before > 0:
        print(f"Improvement      : {total_before / total_after:.2f}x" if total_after > 0 else "N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
