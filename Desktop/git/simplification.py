import argparse
import os
import psutil
import time
from scipy.spatial.distance import directed_hausdorff

from util.mesh import Mesh
def compute_hausdorff(original_vertices, simplified_vertices):
    distance = max(directed_hausdorff(original_vertices, simplified_vertices)[0],
                   directed_hausdorff(simplified_vertices, original_vertices)[0])
    return distance

def get_parser():
    parser = argparse.ArgumentParser(description="Mesh Simplification")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file name")
    parser.add_argument("-v", type=int, help="Target vertex number")
    parser.add_argument("-p", type=float, default=0.5, help="Rate of simplification (Ignored by -v)")
    parser.add_argument("-optim", action="store_true", help="Specify for valence aware simplification")
    parser.add_argument("-isotropic", action="store_true", help="Specify for Isotropic simplification")
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    mesh = Mesh(args.input)
    start_time = time.time()

    mesh_name = os.path.basename(args.input).split(".")[-2]
    if args.v:
        target_v = args.v
    else:
        target_v = int(len(mesh.vs) * args.p)
    if target_v >= mesh.vs.shape[0]:
        print("[ERROR]: Target vertex number should be smaller than {}!".format(mesh.vs.shape[0]))
        exit()
    if args.isotropic:
        simp_mesh = mesh.edge_based_simplification(target_v=target_v, valence_aware=args.optim)
    else:
        simp_mesh = mesh.simplification(target_v=target_v, valence_aware=args.optim)
    os.makedirs("data/output/", exist_ok=True)
    simp_mesh.save("data/output/{}_{}.obj".format(mesh_name, simp_mesh.vs.shape[0]))
    print("[FIN] Simplification Completed!")

    # Compute the Hausdorff distance after simplification
    hausdorff_distance = compute_hausdorff(mesh.vs, simp_mesh.vs)
    print(f"Hausdorff distance: {hausdorff_distance}")
    score = 100 - min(hausdorff_distance, 100)
    print(f"Simplification quality score: {score}")

    end_time = time.time()
    # Memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # in MB

    # Calculate time taken
    execution_time = end_time - start_time

    # Print results
    print(f"Memory usage: {memory_usage} MB")
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()
