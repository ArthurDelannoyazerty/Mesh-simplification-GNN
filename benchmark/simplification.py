import argparse
import os
import time
import numpy as np
from util.mesh import Mesh
from benchmark import performance_metrics
import glob

from stl import mesh
import numpy as np

def convert_stl_to_obj(stl_file_path):
    obj_file_path = stl_file_path.replace('.stl', '.obj')
    stl_mesh = mesh.Mesh.from_file(stl_file_path)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    unique_vertices, faces = np.unique(vertices, axis=0, return_inverse=True)
    faces = faces.reshape(-1, 3) + 1  # OBJ files are 1-indexed
    with open(obj_file_path, 'w') as obj_file:
        for vertex in unique_vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            obj_file.write(f"f {face[0]} {face[1]} {face[2]}\n")
    return obj_file_path

def get_parser():
    parser = argparse.ArgumentParser(description="Mesh Simplification")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory containing .obj files")
    parser.add_argument("-v", type=int, help="Target vertex number")
    parser.add_argument("-optim", action="store_true", help="Specify for valence aware simplification")
    parser.add_argument("-isotropic", action="store_true", help="Specify for Isotropic simplification")
    return parser.parse_args()

def main():
    args = get_parser()
    mesh_files = glob.glob(os.path.join(args.directory, '*.obj')) + glob.glob(os.path.join(args.directory, '*.stl'))
    for mesh_file in mesh_files:
        file_extension = os.path.splitext(mesh_file)[1].lower()
        
        if file_extension == '.stl':
            obj_file = convert_stl_to_obj(mesh_file)
        else:
            obj_file = mesh_file
        
        for rate in np.arange(0.2, 1.2, 0.2):  # Loop over rates from 0.2 to 1.0
            mesh = Mesh(obj_file)
            start_time = time.time()

            mesh_name = os.path.basename(obj_file).split(".")[0]
            target_v = int(len(mesh.vs) * rate)  # Use the rate in the loop

            if target_v >= mesh.vs.shape[0]:
                print(f"[ERROR]: Target vertex number for rate {rate} should be smaller than {mesh.vs.shape[0]}!")
                continue

            if args.isotropic:
                simp_mesh = mesh.edge_based_simplification(target_v=target_v, valence_aware=args.optim)
            else:
                simp_mesh = mesh.simplification(target_v=target_v, valence_aware=args.optim)

            output_dir = "data/output/"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{mesh_name}_{simp_mesh.vs.shape[0]}_rate_{rate:.1f}.obj")
            simp_mesh.save(output_path)
            print(f"[FIN] Simplification Completed for rate {rate:.1f}!")

            end_time = time.time()
            execution_time = end_time - start_time

            # Calculate and print metrics
            metrics = performance_metrics(mesh, simp_mesh, f"data/output/metrics_rate_{rate:.1f}.csv", rate)
            print(f"Execution time for rate {rate:.1f}: {execution_time} seconds")
            print(f"Hausdorff Distance for rate {rate:.1f}: {metrics['hausdorff_distance']}")
            print(f"Chamfer Distance for rate {rate:.1f}: {metrics['chamfer_distance']}")
            print(f"Curvature Error for rate {rate:.1f}: {metrics['curvature_error']}")
            print(f"Memory Usage for rate {rate:.1f}: {metrics['memory_usage']} MB\n")


if __name__ == "__main__":
    main()

