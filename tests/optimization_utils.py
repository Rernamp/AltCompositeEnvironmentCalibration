from lmfit import Parameter, Parameters
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List

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
            cost_func = 1 - value

            
            # cost_func += scale * np.dot(diff_position, diff_position)

            # if (marker_index == 0):
            #     cost_func += scale * np.dot(diff_markers, diff_markers)
            # if (i == 0):
            #     cost_func += scale * np.dot(diff_position, diff_position)

            result.append(cost_func)
    
    for value in diff_markers_from_params.flatten():
        result.append(scale * value)
    
    np_result = np.array(result)
    
    # print(f"Cost {np.sum(np_result)}")
        
    return np_result


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

            grad_by_marker_i = - ((rotated_ray / norm_part) - ((
                dot_part * rotated_ray_norm * pm) / (pm_norm * norm_part * norm_part)))

            grad_by_pos_i = - grad_by_marker_i

            add_point_to_dict(grad_by_name, grad_by_pos_i, f"diff_pos_{i}")
            add_point_to_dict(grad_by_name, grad_by_marker_i, f"diff_marker_{marker_index}")

            grad_by_w_i = -(norm_part * 2 * (np.dot(quat_w * ray + np.cross(quat_u, ray), pm)) - (dot_part / rotated_ray_norm) * (2 * pm_norm * np.dot(rotated_ray, (quat_w * ray + np.cross(quat_u, ray))))) / (norm_part * norm_part)
            grad_by_name[f"quat_w_{i}"] = grad_by_w_i

            grad_by_u_i_dot_part = np.cross(pm, -4 * quat_w * ray - 4 * np.cross(quat_u, ray))  \
                + 2 * quat_w * np.cross(pm, ray) + 2 * np.dot(quat_u, ray) * pm \
                + 2 * quat_u * np.dot(pm, ray) - 2 * ray * np.dot(pm, quat_u)
            grad_by_u_i_norm_part = (pm_norm / rotated_ray_norm) * (np.cross(rotated_ray, (- 4 * quat_w * ray - 4 * np.cross(quat_u, ray))) + \
                + 2 * quat_w * np.cross(rotated_ray, ray) + \
                + 2 * (np.dot(quat_u, ray) * rotated_ray + quat_u * np.dot(rotated_ray, ray)) - \
                - 2 * ray * np.dot(rotated_ray, quat_u)    \
                )

            grad_by_u_i = -(norm_part * grad_by_u_i_dot_part - dot_part * grad_by_u_i_norm_part) / (norm_part * norm_part)
            add_point_to_dict(grad_by_name, grad_by_u_i, f"quat_u_{i}")

            result.append([grad_by_name[name]
                          for name in parameters if parameters[name].vary])

    for i, diff_marker in enumerate(markers_from_params): 
            for coordinate in ["x", "y", "z"]:
                grad_by_name = {
                        parameter: 0 for parameter in parameters.valuesdict()}
                grad_by_name[f"diff_marker_{i}_{coordinate}"]  = scale
                result.append([grad_by_name[name]
                                for name in parameters if parameters[name].vary])
                
        
        # grad_by_name[f"diff_marker_{i}"] = 2 * scale * diff_marker
        
        
    return np.array(result)


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


def generate_offset_variants(points: np.ndarray, offset: np.ndarray, num_variants: int, noise_std: float, generator: np.random.Generator):
    resulted_points = []

    for marker in points:
        for i in range(num_variants):
            random_ray = generator.random(offset.shape)
            random_ray *= noise_std / np.linalg.norm(random_ray)
            resulted_points.append(marker + offset + random_ray)
    return np.array(resulted_points)


def generate_snapshots(positions: np.ndarray, markers: np.ndarray):
    result = []

    for position in positions:
        rays = []
        marker_indices = []
        for marker_index, marker in enumerate(markers):
            ray = marker - position
            marker_indices.append(marker_index)
            rays.append(ray)
        result.append(
            Snapshot(position=position, rays=rays, marker_indices=marker_indices))
    
    return result

def generate_parameters(positions: np.ndarray, markers: np.ndarray):
    parameters = Parameters()
    for i, marker in enumerate(markers):
        add_point_to_parameters(parameters=parameters,
                            point=np.zeros(3), prefix=f"diff_marker_{i}", vary=True)
        
    for i, snapshot in enumerate(positions):
        add_point_to_parameters(parameters=parameters,
                                point=np.zeros(3), prefix=f"diff_pos_{i}", vary=True)

    for i, snapshot in enumerate(positions):
        add_point_to_parameters(parameters=parameters,
                                point=np.zeros(snapshot.position.shape), prefix=f"quat_u_{i}", vary=True)
        parameters.add(Parameter(name=f"quat_w_{i}", value=1, vary=True))
    
    return parameters

@dataclass
class ParametersBuilder:
    snapshots: List[Snapshot]
    markers: np.ndarray
    
    marker_offsets: np.ndarray = field(init=False)
    position_offsets: np.ndarray = field(init=False)

    def __post_init__(self):
        self.marker_offsets = np.zeros(self.markers.shape)
        self.position_offsets = np.zeros([len(self.snapshots), 3])
    
    def add_markers_offsets(self, offsets: np.ndarray):
        assert self.markers.shape == offsets.shape, f"Shape mismatch: {self.markers.shape} vs {offsets.shape}"
        self.marker_offsets = offsets
        
    def add_position_offsets(self, offsets: np.ndarray):
        self.position_offsets = offsets
        
    def getResult(self):
        result = Parameters()
        
        for i, marker in enumerate(self.markers):
            add_point_to_parameters(parameters=result,
                                    point=self.marker_offsets[i, :], prefix=f"diff_marker_{i}", vary=True)
            
        for i, snapshot in enumerate(self.snapshots):
            add_point_to_parameters(parameters=result,
                                    point=np.zeros(3), prefix=f"diff_pos_{i}", vary=True)

        for i, snapshot in enumerate(self.snapshots):
            add_point_to_parameters(parameters=result,
                                    point=self.position_offsets[i, :], prefix=f"quat_u_{i}", vary=False)
            result.add(Parameter(name=f"quat_w_{i}", value=1, vary=False))
            
        return result


def similarity_transform_svd(src, tgt):
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    scale = np.sum(S) / np.sum(src_centered ** 2)
    
    translation = tgt_mean - scale * (src_mean @ R.T)
    
    return scale, R, translation

def recover_target(src_points, tgt_points, query_points=None):
    scale, rotation, translation = similarity_transform_svd(src_points, tgt_points)
    
    if query_points is None:
        query_points = src_points
    
    return scale * (query_points @ rotation.T) + translation