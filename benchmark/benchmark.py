import time
import psutil
import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist
import os


import csv

def append_metrics_to_csv(metrics, filename):
    fieldnames = ['hausdorff_distance', 'chamfer_distance', 'curvature_error', 'execution_time', 'memory_usage', 'simplification_rate']
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header if the file is new

        writer.writerow(metrics)



def read_obj_file(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
    return np.array(vertices)

def compute_hausdorff(original_vertices, simplified_vertices):
    distance = max(directed_hausdorff(original_vertices, simplified_vertices)[0],
                   directed_hausdorff(simplified_vertices, original_vertices)[0])
    return distance

def compute_chamfer_distance(set1, set2):
    d1 = np.mean(np.min(cdist(set1, set2), axis=1))
    d2 = np.mean(np.min(cdist(set2, set1), axis=1))
    return (d1 + d2) / 2

def compute_curvature_error(original_vertices, simplified_vertices):
    # This is a simple placeholder for curvature error
    # You should replace this with a more accurate method if available
    return np.linalg.norm(np.mean(original_vertices, axis=0) - np.mean(simplified_vertices, axis=0))

def performance_metrics(original_mesh_path, simplified_mesh_path, csv_filename,simplification_rate):
    #original_vertices = read_obj_file(original_mesh_path)
    #simplified_vertices = read_obj_file(simplified_mesh_path)

    start_time = time.time()

    hausdorff_distance = compute_hausdorff(original_mesh_path.vs, simplified_mesh_path.vs)
    chamfer_distance = compute_chamfer_distance(original_mesh_path.vs, simplified_mesh_path.vs)
    curvature_error = compute_curvature_error(original_mesh_path.vs, simplified_mesh_path.vs)

    end_time = time.time()
    execution_time = end_time - start_time

    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # in MB

    metrics = {
        "hausdorff_distance": hausdorff_distance,
        "chamfer_distance": chamfer_distance,
        "curvature_error": curvature_error,
        "execution_time": execution_time,
        "memory_usage": memory_usage,
        "simplification_rate": simplification_rate

    }
    # append_metrics_to_csv(metrics, csv_filename)           # saved after
    
    return metrics


