import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
from scipy.optimize import minimize

from EniPy import eniUtils

from PointsCollection import *
from Utils import *

def get_mean_error(points_a, points_b, rotation_b, offset_b):
    errors = []
    for a, b in zip(points_a, points_b):
        transformed = rotation_b.apply(b) + offset_b
        dist = distance.euclidean(a, transformed)
        errors.append(dist)
    errors = np.array(errors)
    return errors.mean()


def f(x, points_a, points_b):
    r = Rotation.from_euler('xyz', angles=[0, x[0], 0], degrees=True)
    return get_mean_error(points_a, points_b, r, x[1:])

points_collection = PointsCollection()
points_collection.import_from_points_collectror(f'dumps/3points.AltPointsCollector.Result.json')

all_points = points_collection.get_all_points()

print(f'detected {len(points_collection.get_environment_names())} environments')
for e in points_collection.get_environment_names():
    print(f'{e}')

print(f'poses_used metrics: {get_metrics([point_info.point.poses_used for point_info in all_points])}')
print(f'position_mean_deviation metrics: {get_metrics([point_info.point.position_mean_deviation for point_info in all_points])}')



expected_points_count = len(points_collection.get_environment_names()) * len(points_collection.get_point_ids()) * len(points_collection.get_alt_indices())
print(f'expected points count: {expected_points_count} actual: {len(all_points)}')

if len(points_collection.get_environment_names()) != 2:
    raise Exception('Currently supported only 2 environment')
if len(points_collection.get_alt_indices()) != 1:
    raise Exception('Currently supported only 1 alt')
if len(all_points) != expected_points_count:
    raise Exception('Mismatching point count')

pointsA = np.empty((0, 3))
pointsB = np.empty((0, 3))



for p in points_collection.get_all_points(points_collection.get_environment_names()[0]):
    pointsA = np.append(pointsA, [p.point.position], 0)
for p in points_collection.get_all_points(points_collection.get_environment_names()[1]):
    pointsB = np.append(pointsB, [p.point.position], 0)

for i in range(len(pointsA) - 1):
    distance_a = distance.euclidean(pointsA[i], pointsA[i + 1])
    distance_b = distance.euclidean(pointsB[i], pointsB[i + 1])
    print(f'distances: {distance_a * 1000:.2f} {distance_b * 1000:.2f} mm diff {(distance_a - distance_b) * 1000:.2f} mm')


base_rotation = 0
base_position = np.array([0, 0, 0])

begin = time.perf_counter()
result = minimize(f, np.array([base_rotation, *base_position]), args=(pointsA, pointsB))
end = time.perf_counter()
print(f'completed by: {((end - begin) * 1000):1f} ms. Result: {result}')
x = result["x"]
print(f'error: {result["fun"] * 1000} mm. rotation: {x[0]} offset: {x[1:]}')

r = Rotation.from_euler('xyz', angles=[0, x[0], 0], degrees=True)
for a, b in zip(pointsA, pointsB):
    e = get_mean_error([a], [b], r, x[1:])
    print(f'{e * 1000:.2f} mm')
