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

def error_between_rays(ray, point_to_marker):
    error = ray - point_to_marker / np.linalg.norm(point_to_marker)
    return np.linalg.norm(error)