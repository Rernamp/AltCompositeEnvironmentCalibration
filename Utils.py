import numpy as np
def get_metrics(collection):
    return {'min': min(collection), 'max': max(collection), 'average': float(np.average(collection))}

def print_table(name, data):
    print(f'{name}')
    for k in sorted(data.keys()):
        str = ''
        for v in sorted(data[k].values()):
            str += f'{v:.2f}\t'
        print(str)

def get_angle_between_quaternions(q0, q1):
    return (q0 * q1.inv()).magnitude()

def get_angle_between_vectors(v0, v1):
    v0_u = v0 / np.linalg.norm(v0)
    v1_u = v1 / np.linalg.norm(v1)
    return np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))