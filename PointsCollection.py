from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import numpy as np

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

    def import_from_points_collectror(self, path):
        raw_dump = eniUtils.readJson(path)
        for tracking_data in raw_dump["AltCollectedTrackingDatas"]:
            environment_index = tracking_data["EnvironmentIndex"]
            point_id = tracking_data["PointIndex"]
            alt_index = tracking_data["NodeProperties"]["sys/HardwareSerialNumber"]
            pose = tracking_data["AltTrackingCombinedPose"]
            p = Point(np.array(pose["Position"]), Rotation.from_quat(pose["Rotation"]), pose["PosesUsed"],
                      pose["PositionMeanDeviation"])
            self.add_point(environment_index, point_id, alt_index, p)

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
        return [*self._collection]

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