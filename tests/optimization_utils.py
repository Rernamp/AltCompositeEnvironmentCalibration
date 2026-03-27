from lmfit import Parameter, Parameters
import numpy as np
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Utils import *


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
            
            cost_func += scale * np.dot(diff_markers, diff_markers)
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
            
            grad_by_marker_i += 2 * scale * markers_from_params[marker_index]
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


# def calc_position_by_markers()