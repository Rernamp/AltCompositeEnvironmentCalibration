from lmfit import Parameters, minimize, report_fit
import numpy as np

def residuals(parameters, points, phi):
    result = np.zeros((len(phi), 3))

    R = parameters["R"].value
    x0 = parameters["x0"].value
    y0 = parameters["y0"].value
    B = parameters["B"].value

    for index, angle in enumerate(phi):
        result[index] = np.array([R * np.cos(angle) + x0, R * np.sin(angle) + y0, B])
        result[index] = result[index] - points[index]

    return result

x0 = 1.5
y0 = 1
R = 2
B = 1

params = Parameters()
params.add("x0", value=1.5)
params.add("y0", value=1.5)
params.add("R", value=1.5)
params.add("B", value = 0.1)

phi = np.linspace(0, 360)

points = np.zeros((len(phi), 3))

for index, angle in enumerate(phi):
    points[index] = np.array([R * np.cos(angle) + x0, R * np.sin(angle) + y0, B]) + np.random.uniform(-1, 1)

result = minimize(residuals, params, args=[points, phi])

print(result.params)