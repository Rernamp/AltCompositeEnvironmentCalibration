from scipy.spatial import distance

from PointsCollection import *
from Utils import *



points_collection = PointsCollection()
points_collection.import_from_points_collectror(f'dumps/8points.AltPointsCollector.Result.json')

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
