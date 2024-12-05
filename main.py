import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
from scipy.optimize import minimize

from EniPy import eniUtils


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


dump_points = eniUtils.readJson(f'dumps/OneAlt.ManualOffsetSamples.Phi.json')

pointsA = np.empty((0, 3))
pointsB = np.empty((0, 3))


for p in dump_points:
    environment_poses = p['EnvironmentPoses']
    if len(environment_poses) != 2:
        raise Exception('Currently supported only 2 environment')
    pointsA = np.append(pointsA, [environment_poses[0]["Position"]], 0)
    pointsB = np.append(pointsB, [environment_poses[1]["Position"]], 0)

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
