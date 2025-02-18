import numpy as np

from EniPy import eniUtils

Step = 4
XSize = 3
ZSize = 3

StartX = -1.5
StartZ = -1.5
Height = 2
Rotation = np.array([0.7071068, 0, 0, 0.7071068])
StandingTimeSeconds = 0.1
ToNextPointTimeSeconds = 1

points = []
for i in range(Step + 1):
    for j in range(Step + 1):
        p = {}
        x = StartX + (i * XSize / Step)
        if (i % 2 == 0):
            z = StartZ + (j * ZSize / Step)
        else:
            z = StartZ + ((Step - j) * ZSize / Step)
        p["Position"] = np.array([x, Height, z])
        p["Rotation"] = Rotation
        p["StandingTimeSeconds"] = StandingTimeSeconds
        p["ToNextPointTimeSeconds"] = ToNextPointTimeSeconds
        points.append(p)

eniUtils.writeJson(f'trajectory.json', points)