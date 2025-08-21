from Utils import read_dump
import numpy as np
import matplotlib.pyplot as plt

def point_to_str(point):
    return f"{point[0]}, {point[1]}, {point[2]}"

if __name__ == "__main__":
    guess, snapshots, markers = read_dump("dataset/allAxis_1Marker_fix3/#000.json")

    scale = 2

    colorize_point = []

    with open('geogebraScript.txt', "w") as f:
        f.write(f"scale={scale}\n")
        for marker_index, marker in enumerate(markers):
            f.write(f"M_{marker_index}=({point_to_str(marker)}) * scale\n");

        for i, snapshot in enumerate(snapshots):
            position = snapshot.position
            f.write(f"P_{i}=({point_to_str(position)}) * scale\n")

            colorize_point.append(f"SetColor(P_{i}, 0, 1, 0)\n")
            for j, marker_index in enumerate(snapshot.marker_indices):
                marker = markers[marker_index]
                segmentName = f"PtoM_{i}_{marker_index}"
                f.write(f"{segmentName} = Segment(P_{i}, M_{marker_index})\n")
                colorize_point.append(f"SetColor({segmentName}, 0, 0, 1)\n")

            for j, ray in enumerate(snapshot.rays):
                f.write(f"R_{i}_{j} = Segment(P_{i},({point_to_str(ray)}) + P_{i})\n")
                colorize_point.append(f"SetColor(R_{i}_{j}, 1, 0, 0)\n")

        for color in colorize_point:
            f.write(color)

    # snapshot = snapshots[0]
    #
    # markers_in_snapshot = np.array([np.array(markers[marker_index]) for marker_index in snapshot.marker_indices])
    #
    # positions_for_markers = np.array([np.array(snapshot.position) for index in snapshot.marker_indices])
    #
    # x, y, z = np.split(positions_for_markers, 3, axis=1)
    #
    # u, v, w = np.split(markers_in_snapshot - positions_for_markers, 3, axis=1)
    #
    # uR, vR, wR = np.split(np.array(snapshot.rays), 3, axis=1)
    #
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # ax.quiver(x, y, z, u, v, w)
    # ax.quiver(x, y, z, u, v, w, color="blue")
    # ax.quiver(x, y, z, uR, vR, wR, color="red")
    #
    # ax.set(xticklabels=[],
    #        yticklabels=[],
    #        zticklabels=[])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # ax.set_xlim(min(min(x), min(u)), max(max(x), max(u)))
    # ax.set_ylim(min(min(y), min(v)), max(max(y), max(v)))
    # ax.set_zlim(min(min(z), min(w)), max(max(z), max(w)))
    #
    # plt.show()



