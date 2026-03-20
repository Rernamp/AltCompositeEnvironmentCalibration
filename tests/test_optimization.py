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
        raise ValueError(f"num_elements ({num_elements}) должно быть полным квадратом")
    
    offset = (side - 1) * step / 2
    
    x_coords = np.arange(side) * step - offset
    y_coords = np.arange(side) * step - offset
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(num_elements)])
    
    return points

markers = square_grid_points_centered(0.1, 9)

offset = 1

positions = []

points_variants = 10

for marker in markers:
    positions.append(marker + np.array([0, 0, offset]))
    for i in range(points_variants):
        positions.append(marker + np.array([np.random.randn(), np.random.randn(), offset]))
positions = np.array(positions)

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


max_offset = 0.01

for i, snapshot in enumerate(modified_snapshots):
    snapshot.position = snapshot.position + \
        np.random.rand(snapshot.position.shape[0]) * max_offset
    # snapshot.position = snapshot.position + np.array([1, 1, 1]) * max_offset
    print(f"Modify position {i} {snapshot.position}")

parameters = Parameters()

snapshot_for_optimization = modified_snapshots


for i, marker in enumerate(markers):
    # modify_marker = marker
    modify_marker = marker + \
        np.random.rand(marker.shape[0]) * max_offset
    add_point_to_parameters(parameters=parameters,
                            point=modify_marker, prefix=f"marker_{i}", vary=True)
    
for i, snapshot in enumerate(snapshot_for_optimization):
    add_point_to_parameters(parameters=parameters,
                            point=snapshot.position, prefix=f"pos_{i}", vary=(i != 0))

    
for i, snapshot in enumerate(snapshot_for_optimization):
    add_point_to_parameters(parameters=parameters,
                            point=np.zeros(snapshot.position.shape), prefix=f"quat_u_{i}", vary=(i != 0))
    parameters.add(Parameter(name=f"quat_w_{i}", value=1, vary=(i != 0)))


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
        
        quat_u = extract_point_from_parameters(parameters=parameters, prefix=f"quat_u_{i}")
        quat_w = parameters[f"quat_w_{i}"]
        for ray_it, ray in enumerate(snapshot.rays):
            grad_by_name = {
                parameter: 0 for parameter in parameters.valuesdict()}

            rotated_ray = quat_w * quat_w * ray + 2 * (quat_w * np.cross(quat_u, ray) + quat_u * np.dot(quat_u, ray)) - np.dot(quat_u, quat_u) * ray
            
            rotated_ray_norm = np.linalg.norm(rotated_ray)
            marker_index = snapshot.marker_indices[ray_it]
            pm = markers[marker_index] - position
            pm_norm = np.linalg.norm(pm)
            
            dot_part = np.dot(rotated_ray, pm)
            norm_part = pm_norm * rotated_ray_norm
            
            cost_func = dot_part / norm_part
            
            grad_by_pos_i = (2 * cost_func) * ((rotated_ray / norm_part) - (
                dot_part * rotated_ray_norm * pm) / (pm_norm * norm_part * norm_part))
            grad_by_marker_i = - grad_by_pos_i
            
            
            add_point_to_dict(grad_by_name, grad_by_pos_i, f"pos_{i}")
            add_point_to_dict(grad_by_name, grad_by_marker_i, f"marker_{marker_index}")
            
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


cost_func_before = cost_func(
    parameters=parameters, snapshots=snapshot_for_optimization, markers=markers)

mini = Minimizer(cost_func, parameters, fcn_args=(
    synthetic_snapshots, markers))
result = mini.minimize(method='leastsq', **
                       {'Dfun': gradient_function, "gtol": 1e-14})
# result = mini.minimize(method='leastsq')
# result = mini.minimize(method='least_squares', jac=gradient_function)

with open('output.txt', 'w') as f:
    f.write(fit_report(result))
    

cost_func_after = cost_func(
    parameters=result.params, snapshots=snapshot_for_optimization, markers=markers)

print(f"Before optimization cost func value: {cost_func_before}")
print(f"After optimization cost func value: {cost_func_after}")


print(f"Func call number: {result.nfev}")

# print(gradient_function(parameters=result.params, snapshots=snapshot_for_optimization, markers=markers))

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

for i, synt_positional in enumerate(positions):
    start_position = extract_point_from_parameters(parameters=parameters, prefix=f"pos_{i}")
    optimized_position = extract_point_from_parameters(parameters=result.params, prefix=f"pos_{i}")
    
    print(f"Position {i}; True : {synt_positional}; Optimized: {optimized_position}; Start: {start_position}")

for i, synt_marker in enumerate(markers):
    start_marker = extract_point_from_parameters(parameters=parameters, prefix=f"marker_{i}")
    optimized_marker = extract_point_from_parameters(parameters=result.params, prefix=f"marker_{i}")
    
    print(f"Marker {i}; True : {synt_marker}; Optimized: {optimized_marker}; Start: {start_marker}")