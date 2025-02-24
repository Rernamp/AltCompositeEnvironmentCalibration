import numpy as np
import subprocess
import copy
from EniPy import eniUtils

def generate_trajectory():
    Step = 3
    XSize = 3
    ZSize = 3

    StartX = -1.5
    StartZ = -1.5
    Height = 2
    Rotation = np.array([90, 0, 0])
    StandingTimeSeconds = 0.1
    ToNextPointTimeSeconds = 0.5

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
            p["RotationEulerAngles"] = Rotation
            p["StandingTimeSeconds"] = StandingTimeSeconds
            p["ToNextPointTimeSeconds"] = ToNextPointTimeSeconds
            points.append(p)
    return points


def generate_environment_code(basic_code, offsets):
    input_path = f'input.json'
    output_path = f'output.json'

    input = {}
    input["LogLevel"] = "Info"
    e = {}
    e["code"] = basic_code
    e["position"] = np.array([0, 0, 0])
    e["rotation"] = np.array([0, 0, 0])
    e["minRaysForMatch"] = 3
    e["minRaysForMatchByPosition"] = 3
    e["markersOffsets"] = offsets
    input["UnderlyingEnvironments"] = [e]

    eniUtils.writeJson(input_path, input)
    subprocess.run([R"D:\FactoryUtilities\AntilatencyAltEnvironmentComposite.Editor\AntilatencyAltEnvironmentComposite.Editor.exe", "V1", input_path, output_path])
    result = eniUtils.readJson(output_path)
    return result["code"]

dataset = []

no_offsets = [np.array([0.0, 0.0, 0.0]) for _ in range(12)]
for i in range(3):
    offsets = copy.deepcopy(no_offsets)
    offsets[i][0] = 0.01
    code = generate_environment_code("AntilatencyAltEnvironmentHorizontalGrid~AgAEBLhTiT_cRqA-r45jvZqZmT4AAAAAAAAAAACamRk_AQQCAwIDAgMCAAIAAgM", offsets)
    dataset.append({"offsets" : offsets, "code" : code})

snake_trajectory = generate_trajectory()

simulator_config = {}
rotated_last_point = copy.deepcopy(snake_trajectory[-1])
rotated_last_point["RotationEulerAngles"][0] -= 180
simulator_config["TrajectoryPoints"] = [*snake_trajectory, rotated_last_point]
commands = []
for sample in dataset:
    commands.append(f"SetEnvironment {sample["code"]}")
    commands.append("Move")
simulator_config["Commands"] = commands
eniUtils.writeJson(f'simulator_config.json', simulator_config)