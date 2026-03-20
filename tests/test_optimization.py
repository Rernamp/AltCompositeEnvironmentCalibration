from lmfit import Parameter, Parameters, Minimizer, minimize, report_fit
import numpy as np
import copy
import sys
from random import randrange
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *


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
    # snapshot.position = snapshot.position + np.array([1, 1, 1]) * max_offset
    print(f"Modify position {i} {snapshot.position}")

parameters = Parameters()

snapshot_for_optimization = modified_snapshots

for i, snapshot in enumerate(snapshot_for_optimization):
    add_point_to_parameters(parameters=parameters,
                            point=snapshot.position, prefix=f"pos_{i}")

    add_point_to_parameters(parameters=parameters,
                            point=np.zeros(snapshot.position.shape), prefix=f"quat_u_{i}", vary=False)
    parameters.add(Parameter(name=f"quat_w_{i}", value=1, vary=False))

for i, marker in enumerate(markers):
    modify_marker = marker
    # modify_marker = marker + \
    #     np.random.rand(marker.shape[0]) * max_offset
    add_point_to_parameters(parameters=parameters,
                            point=modify_marker, prefix=f"marker_{i}", vary=False)


def cost_func(parameters: Parameters, snapshots: [Snapshot], markers):
    result = []

    for i, snapshot in enumerate(snapshots):

        markers_from_params = np.array([extract_point_from_parameters(
            parameters=parameters, prefix=f"marker_{marker_index}") for marker_index in snapshot.marker_indices])
        position = extract_point_from_parameters(
            parameters=parameters, prefix=f"pos_{i}")
        
        quat_u = extract_point_from_parameters(parameters=parameters, prefix=f"quat_u_{i}")
        quat_w = parameters[f"quat_w_{i}"]
        for ray_it, ray in enumerate(snapshot.rays):
            rotated_ray = quat_w * quat_w * ray + 2 * (quat_w * np.cross(quat_u, ray) + quat_u * np.dot(quat_u, ray)) - np.dot(quat_u, quat_u) * ray
            pm = markers_from_params[snapshot.marker_indices[ray_it], :] - position
            dot_part = np.dot(rotated_ray, pm)
            norm_part = np.linalg.norm(pm) * np.linalg.norm(rotated_ray)
            value = dot_part / norm_part
            result.append(1 - (value * value))
    return np.array(result)


def gradient_function(parameters: Parameters, snapshots: [Snapshot], markers):
    result = []
    for i, snapshot in enumerate(snapshots):
        position = extract_point_from_parameters(
            parameters=parameters, prefix=f"pos_{i}")
        for ray_it, ray in enumerate(snapshot.rays):
            grad_by_name = {
                parameter: 0 for parameter in parameters.valuesdict()}

            marker_index = snapshot.marker_indices[ray_it]
            pm = markers[marker_index] - position
            dot_part = np.dot(ray, pm)
            norm_part = np.linalg.norm(pm) * np.linalg.norm(ray)
            cost_func = dot_part / norm_part
            grad_by_pos_i = (2 * cost_func) * ((ray / norm_part) - (
                dot_part * np.linalg.norm(ray) * pm) / (np.linalg.norm(pm) * norm_part * norm_part))
            grad_by_marker_i = - grad_by_pos_i
            add_point_to_dict(grad_by_name, grad_by_pos_i, f"pos_{i}")
            add_point_to_dict(grad_by_name, grad_by_marker_i, f"marker_{marker_index}")
            result.append([grad_by_name[name]
                          for name in parameters if parameters[name].vary])

    return np.array(result)


cost_func_before = cost_func(
    parameters=parameters, snapshots=snapshot_for_optimization, markers=markers)

mini = Minimizer(cost_func, parameters, fcn_args=(
    synthetic_snapshots, markers))
result = mini.minimize(method='leastsq', **
                       {'Dfun': gradient_function, "gtol": 1e-10})
# result = mini.minimize(method='leastsq')
# result = mini.minimize(method='least_squares', jac=gradient_function)

print(report_fit(result))

cost_func_after = cost_func(
    parameters=result.params, snapshots=snapshot_for_optimization, markers=markers)

print(f"Before optimization cost func value: {cost_func_before}")
print(f"After optimization cost func value: {cost_func_after}")


print(f"Func call number: {result.nfev}")

# print(gradient_function(parameters=result.params, snapshots=snapshot_for_optimization, markers=markers))
