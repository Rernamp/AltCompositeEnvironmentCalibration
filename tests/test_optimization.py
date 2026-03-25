from lmfit import Parameter, Parameters, Minimizer, minimize, report_fit, fit_report
import numpy as np
import copy
import sys
from random import randrange
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *

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

markers = square_grid_points_centered(0.1, 9)

offset = 1

positions = []

points_variants = 4

for marker in markers:
    # positions.append(marker + np.array([0, 0, offset]))
    for i in range(points_variants):
        positions.append(marker + np.array([np.random.randn(), np.random.randn(), offset]))
positions = np.array(positions)

print(f"Markers: {markers}")
print(f"Positions: {positions}")

synthetic_snapshots = []

for i, position in enumerate(positions):
    rays = []
    marker_indices = []
    for marker_index, marker in enumerate(markers):
        ray = marker - position
        marker_indices.append(marker_index)
        rays.append(ray)
    synthetic_snapshots.append(
        Snapshot(position=position, rays=rays, marker_indices=marker_indices))

modified_snapshots = copy.deepcopy(synthetic_snapshots)

max_offset = 0.01

for i, snapshot in enumerate(modified_snapshots):
    if i >= 1:
        snapshot.position = snapshot.position + \
            np.random.rand(snapshot.position.shape[0]) * max_offset
    # snapshot.position = snapshot.position + \
    #         np.random.rand(snapshot.position.shape[0]) * max_offset
    print(f"Modify position {i} {snapshot.position}")

parameters = Parameters()

snapshot_for_optimization = modified_snapshots

modify_markers = []
for i, marker in enumerate(markers):
    max_marker_offset = 0.01
    
    modify_marker = marker
    if i >= 1:
        modify_marker = marker + \
            np.random.rand(marker.shape[0]) * max_marker_offset
    
    # modify_marker = marker + \
    #     np.random.rand(marker.shape[0]) * max_marker_offset
    add_point_to_parameters(parameters=parameters,
                            point=np.zeros(3), prefix=f"diff_marker_{i}", vary=(i != 0))
    modify_markers.append(modify_marker)
    
for i, snapshot in enumerate(snapshot_for_optimization):
    add_point_to_parameters(parameters=parameters,
                            point=np.zeros(3), prefix=f"diff_pos_{i}", vary=(i != 0))

for i, snapshot in enumerate(snapshot_for_optimization):
    add_point_to_parameters(parameters=parameters,
                            point=np.zeros(snapshot.position.shape), prefix=f"quat_u_{i}", vary=False)
    parameters.add(Parameter(name=f"quat_w_{i}", value=1, vary=False))


def cost_func(parameters: Parameters, snapshots: [Snapshot], markers, scale: float):
    result = []

    diff_markers_from_params = np.array([extract_point_from_parameters(
        parameters=parameters, prefix=f"diff_marker_{i}") for i, marker in enumerate(markers)])
    for i, snapshot in enumerate(snapshots):
        diff_position = extract_point_from_parameters(
            parameters=parameters, prefix=f"diff_pos_{i}")
        position = snapshot.position + diff_position
        
        quat_u = extract_point_from_parameters(parameters=parameters, prefix=f"quat_u_{i}")
        quat_w = parameters[f"quat_w_{i}"]
        for ray_it, ray in enumerate(snapshot.rays):
            rotated_ray = quat_w * quat_w * ray + 2 * (quat_w * np.cross(quat_u, ray) + quat_u * np.dot(quat_u, ray)) - np.dot(quat_u, quat_u) * ray
            marker_index = snapshot.marker_indices[ray_it]
            diff_markers = diff_markers_from_params[marker_index, :]
            pm = markers[marker_index] + diff_markers - position
            dot_part = np.dot(rotated_ray, pm)
            norm_part = np.linalg.norm(pm) * np.linalg.norm(rotated_ray)
            value = dot_part / norm_part
            cost_func = 1 - value * value
            
            # cost_func += scale * np.dot(diff_markers, diff_markers)
            # cost_func += scale * np.dot(diff_position, diff_position)
            
            # if (marker_index == 0):
            #     cost_func += scale * np.dot(diff_markers, diff_markers)
            # if (i == 0):
            #     cost_func += scale * np.dot(diff_position, diff_position)
            
            result.append(cost_func)
    return np.array(result)


def gradient_function(parameters: Parameters, snapshots: [Snapshot], markers, scale: float):
    result = []
    markers_from_params = np.array([extract_point_from_parameters(
        parameters=parameters, prefix=f"diff_marker_{i}") for i, marker in enumerate(markers)])
    for i, snapshot in enumerate(snapshots):
        diff_pos = extract_point_from_parameters(
            parameters=parameters, prefix=f"diff_pos_{i}")
        position = snapshot.position + diff_pos
        
        quat_u = extract_point_from_parameters(parameters=parameters, prefix=f"quat_u_{i}")
        quat_w = parameters[f"quat_w_{i}"]
        for ray_it, ray in enumerate(snapshot.rays):
            grad_by_name = {
                parameter: 0 for parameter in parameters.valuesdict()}

            rotated_ray = quat_w * quat_w * ray + 2 * (quat_w * np.cross(quat_u, ray) + quat_u * np.dot(quat_u, ray)) - np.dot(quat_u, quat_u) * ray
            
            rotated_ray_norm = np.linalg.norm(rotated_ray)
            marker_index = snapshot.marker_indices[ray_it]
            pm = markers[marker_index] + markers_from_params[marker_index] - position
            pm_norm = np.linalg.norm(pm)
            
            dot_part = np.dot(rotated_ray, pm)
            norm_part = pm_norm * rotated_ray_norm
            
            cost_func = dot_part / norm_part
            
            grad_by_pos_i = (2 * cost_func) * ((rotated_ray / norm_part) - (
                dot_part * rotated_ray_norm * pm) / (pm_norm * norm_part * norm_part))
            
            grad_by_marker_i = - grad_by_pos_i
            
            # grad_by_marker_i += 2 * scale * markers_from_params[marker_index]
            # grad_by_pos_i += 2 * scale * diff_pos
            
            # if (marker_index == 0):
            #     grad_by_marker_i += 2 * scale * markers_from_params[marker_index]
            # if (i == 0):
            #     grad_by_pos_i += 2 * scale * diff_pos
            
            add_point_to_dict(grad_by_name, grad_by_pos_i, f"diff_pos_{i}")
            add_point_to_dict(grad_by_name, grad_by_marker_i, f"diff_marker_{marker_index}")
            
            grad_by_w_i = (norm_part * 2 * (np.dot(quat_w * ray + np.cross(quat_u, ray), pm)) - (dot_part / rotated_ray_norm) * (2 * pm_norm * np.dot(rotated_ray, (quat_w * ray + np.cross(quat_u, ray))))) / (norm_part * norm_part)
            grad_by_name[f"quat_w_{i}"] = grad_by_w_i
            
            grad_by_u_i_dot_part = np.cross(pm, -4 * quat_w * ray - 4 * np.cross(quat_u, ray))  \
                + 2 * quat_w * np.cross(pm, ray) + 2 * np.dot(quat_u, ray) * pm \
                + 2 * quat_u * np.dot(pm, ray) - 2 * ray * np.dot(pm, quat_u)
            grad_by_u_i_norm_part = (pm_norm / rotated_ray_norm) * (np.cross(rotated_ray, (- 4 * quat_w * ray - 4 * np.cross(quat_u, ray))) + \
                + 2 * quat_w * np.cross(rotated_ray, ray) + \
                + 2 * (np.dot(quat_u, ray) * rotated_ray + quat_u * np.dot(rotated_ray, ray)) - \
                - 2 * ray * np.dot(rotated_ray, quat_u)    \
                )
            
            grad_by_u_i = (norm_part * grad_by_u_i_dot_part - dot_part * grad_by_u_i_norm_part) / (norm_part * norm_part)
            add_point_to_dict(grad_by_name, grad_by_u_i, f"quat_u_{i}")
            
            result.append([grad_by_name[name]
                          for name in parameters if parameters[name].vary])

    return np.array(result)

scale = 0.4
cost_func_synt = cost_func(
    parameters=parameters, snapshots=synthetic_snapshots, markers=markers, scale=scale)

cost_func_before = cost_func(
    parameters=parameters, snapshots=snapshot_for_optimization, markers=modify_markers, scale=scale)

mini = Minimizer(cost_func, parameters, fcn_args=(
    snapshot_for_optimization, modify_markers, scale))

result = mini.minimize(method='leastsq', **
                       {'Dfun': gradient_function, 
                        "gtol": 1e-16,
                        "ftol": 1e-16})
# result = mini.minimize(method='leastsq', **
#                        {'Dfun': gradient_function})
# result = mini.minimize(method='leastsq')
# result = mini.minimize(method='least_squares', jac=gradient_function)

with open('output.txt', 'w') as f:
    f.write(fit_report(result))
    

cost_func_after = cost_func(
    parameters=result.params, snapshots=snapshot_for_optimization, markers=modify_markers, scale=scale)

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

for i, synt_marker in enumerate(markers):
    start_marker_diff = extract_point_from_parameters(parameters=parameters, prefix=f"diff_marker_{i}")
    optimized_marker_diff = extract_point_from_parameters(parameters=result.params, prefix=f"diff_marker_{i}")
    
    modify_marker = modify_markers[i]
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

print(f"Grad func in synt data {np.sum(np.sum(gradient_function(parameters=parameters, snapshots=synthetic_snapshots, markers=markers, scale=scale)))}")