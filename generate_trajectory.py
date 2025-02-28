import numpy as np
import subprocess
import copy
import random
from EniPy import eniUtils

def execute_composite_editor(mode, input_path, output_path):
    subprocess.run([R"D:\FactoryUtilities\AntilatencyAltEnvironmentComposite.Editor\AntilatencyAltEnvironmentComposite.Editor.exe", mode, input_path, output_path])

def generate_trajectory():
    Step = 3
    XSize = 3
    ZSize = 3

    StartX = -1.5
    StartZ = -1.5
    Height = 2
    Rotation = np.array([90, 0, 0])
    StandingTimeSeconds = 0.1
    ToNextPointTimeSeconds = 0.2

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
    execute_composite_editor("V1", input_path, output_path)
    result = eniUtils.readJson(output_path)
    return result["code"]

dataset = []
markers_count = 12
# one axis, 1 marker, [+10mm, -10mm]

# one axis, 2 markers

# all axis, 1 marker, [(+10, -5, +15), (15, 10, 10), (-5, 5, 15)]
no_offsets = [np.array([0.0, 0.0, 0.0]) for _ in range(markers_count)]
# fixed_errors = [[+10, -5, +15], [15, 10, 10], [-5, 5, 15]]
# for fixed_error in fixed_errors:
#     for marker_index in range(markers_count):
#         offsets = copy.deepcopy(no_offsets)
#         offsets[marker_index] = np.array([v / 1000.0 for v in fixed_error])
#         code = generate_environment_code("AntilatencyAltEnvironmentHorizontalGrid~AgAEBLhTiT_cRqA-r45jvZqZmT4AAAAAAAAAAACamRk_AQQCAwIDAgMCAAIAAgM", offsets)
#         dataset.append({"offsets" : offsets, "code" : code})

random_variants = [0, -2, 2, -5, 5, -10, 10, -15, 15]
for _ in range(10):
    offsets = copy.deepcopy(no_offsets)
    for marker_index in range(markers_count):
        offsets[marker_index] = np.array([random.choice(random_variants) / 1000.0 for _ in range(3)])
    code = generate_environment_code("AntilatencyAltEnvironmentHorizontalGrid~AgAEBLhTiT_cRqA-r45jvZqZmT4AAAAAAAAAAACamRk_AQQCAwIDAgMCAAIAAgM", offsets)
    dataset.append({"offsets" : offsets, "code" : code})

snake_trajectory = generate_trajectory()

simulator_config = {}
rotated_last_point = copy.deepcopy(snake_trajectory[-1])
rotated_last_point["RotationEulerAngles"][0] -= 180
rotated_last_point["Position"][0] = 0
rotated_last_point["Position"][2] = 0
simulator_config["TrajectoryPoints"] = [*snake_trajectory, rotated_last_point]
commands = []
for sample in dataset:
    commands.append(f"SetEnvironment {sample["code"]}")
    commands.append("Move")
simulator_config["Commands"] = commands
eniUtils.writeJson(f'simulator_config.json', simulator_config)
eniUtils.writeJson(f'dataset_base.json', dataset)

rays_viewer_dump = input(f'Write path to AltRaysViewer dump that contains recording of playing current config: ')
temp_input_json = f'temp_input.json'
temp_output_json = f'temp_output.json'
eniUtils.writeJson(temp_input_json, {"AltRaysViewerDumpPath": rays_viewer_dump})
execute_composite_editor("MarkersOffsetCalibration", temp_input_json, temp_output_json)
dumps = eniUtils.readJson(temp_output_json)["dumps"]
if len(dumps) != len(dataset):
    print(f'Mismatch length of dumps in RaysViewer and dataset. {len(dumps)} != {len(dataset)}')
    exit(1)
for dump, data in zip(dumps, dataset):
    data["dump"] = dump

for index, data in enumerate(dataset):
    eniUtils.writeJson(f'dataset/full_random/#{index:03d}.json', data)