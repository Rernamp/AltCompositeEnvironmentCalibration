import numpy as np
import glob
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from scipy.spatial import distance

from EniPy import eniUtils


@dataclass
class Point:
    position: np.ndarray
    rotation: Rotation
    poses_used: int
    position_mean_deviation: float

@dataclass
class PointInfo:
    point: Point
    environment_name: str
    point_id: str
    alt_index: int


class PointsCollection:

    def __init__(self):
        self._collection = {}

    def add_point(self, environment_name, point_id, alt_index, point: Point):
        if environment_name not in self._collection:
            self._collection[environment_name] = {}
        env = self._collection[environment_name]
        if point_id not in env:
            env[point_id] = {}
        points = env[point_id]
        if alt_index not in points:
            points[alt_index] = {}
        points[alt_index] = point

    def get_environment_names(self):
        return self._collection.keys()

    def get_point_ids(self):
        result = set()
        for (environment_name, points) in self._collection.items():
            result.update(points.keys())
        return result

    def get_alt_indices(self):
        result = set()
        for (environment_name, points) in self._collection.items():
            for (point_id, alts) in points.items():
                result.update(alts.keys())
        return result

    def get_all_points(self, environment_name = None, point_id = None):
        result = []
        target_environments = self._collection.items()
        if environment_name is not None:
            target_environments = [(environment_name, self._collection[environment_name])]
        for (env, points) in target_environments:
            target_points = points.items()
            if point_id is not None:
                target_points = [(point_id, points[point_id])]
            for (p_id, alts) in target_points:
                for (alt_index, point) in alts.items():
                    result.append(PointInfo(point, env, p_id, alt_index))
        return result

def get_metrics(collection):
    return {'min': min(collection), 'max': max(collection), 'average': float(np.average(collection))}
def read_points(path):
    result = PointsCollection()
    for filename in glob.glob(f'{path}/*_*.json'):
        p = Path(filename)
        point_id, environment_name = p.stem.split("_")

        raw_dumps = eniUtils.readJson(filename)

        for i, raw_data in enumerate(raw_dumps):
            environment_poses = raw_data['EnvironmentPoses']
            if len(environment_poses) != 1:
                raise Exception(f'Should be only 1 environment in {filename}')
            pose = environment_poses[0]
            p = Point(np.array(pose["Position"]), Rotation.from_quat(pose["Rotation"]), pose["PosesUsed"],
                      pose["PositionMeanDeviation"])
            result.add_point(environment_name, point_id, i, p)

    return result

def print_table(name, data):
    print(f'{name}')
    for k in sorted(data.keys()):
        str = ''
        for v in sorted(data[k].values()):
            str += f'{v:.2f}\t'
        print(str)

def get_angle_between_quaternions(q0, q1):
    return (q0 * q1.inv()).magnitude()

points_collection = read_points(f'static_dumps/one_point_exposure_diff/')
all_points = points_collection.get_all_points()

print(f'detected {len(points_collection.get_environment_names())} environments')
for e in points_collection.get_environment_names():
    print(f'{e}')

print(f'poses_used metrics: {get_metrics([point_info.point.poses_used for point_info in all_points])}')
print(f'position_mean_deviation metrics: {get_metrics([point_info.point.position_mean_deviation for point_info in all_points])}')

expected_points_count = len(points_collection.get_environment_names()) * len(points_collection.get_point_ids()) * len(points_collection.get_alt_indices())
print(f'expected points count: {expected_points_count} actual: {len(all_points)}')


for environment_name in points_collection.get_environment_names():
    position_error_table = {}
    rotation_error_table = {}
    for point_id in points_collection.get_point_ids():
        alt_points = points_collection.get_all_points(environment_name, point_id)
        mean_position = np.array([alt_point.point.position for alt_point in alt_points]).mean(axis=0)
        mean_rotation = Rotation([alt_point.point.rotation.as_quat() for alt_point in alt_points]).mean()

        # mean_position = alt_points[0].point.position
        # mean_rotation = alt_points[0].point.rotation

        for alt_point in alt_points:
            if alt_point.alt_index not in position_error_table:
                position_error_table[alt_point.alt_index] = {}
            position_error_table[alt_point.alt_index][alt_point.point_id] = distance.euclidean(mean_position, alt_point.point.position) * 1000

            if alt_point.alt_index not in rotation_error_table:
                rotation_error_table[alt_point.alt_index] = {}
            rotation_error_table[alt_point.alt_index][alt_point.point_id] = np.rad2deg((alt_point.point.rotation * mean_rotation.inv()).magnitude())
    print_table(f'position errors for {environment_name}', position_error_table)
    print_table(f'rotation errors for {environment_name}', rotation_error_table)
