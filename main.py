import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
from scipy.optimize import minimize


from EniPy import eniUtils

def get_mean_error(pointsA, pointsB, rotation_b, offset_b):
    errors = []
    for a, b in zip(pointsA, pointsB):
        tB = rotation_b.apply(b) + offset_b
        dist = distance.euclidean(a, tB)
        errors.append(dist)
    errors = np.array(errors)
    return errors.mean()

def f(x, pointsA, pointsB):
    r = Rotation.from_euler('xyz', angles=[0, x[0], 0], degrees=True)
    return get_mean_error(pointsA, pointsB, r, x[1:])

dumpPoints = eniUtils.readJson(f'dumps/5point.json')

pointsA = np.empty((0, 3))
pointsB = np.empty((0, 3))

for p in dumpPoints:
    environmentPoses = p['EnvironmentPoses']
    if len(environmentPoses) != 2:
        raise Exception('Cuurently supported only 2 environment')
    pointsA = np.append(pointsA, [environmentPoses[0]["Position"]], 0)
    pointsB = np.append(pointsB, [environmentPoses[1]["Position"]], 0)


base_rotation = 179.74
base_position = np.array([-0.087, 0.036, -0.35])

begin = time.perf_counter()
result = minimize(f, [base_rotation, *base_position], args=(pointsA, pointsB))
end = time.perf_counter()
print(f'completed by: {((end - begin) * 1000):1f} ms. Result: {result}')
x = result["x"]
print(f'error: {result["fun"] * 1000} mm. rotation: {x[0]} offset: {x[1:]}')