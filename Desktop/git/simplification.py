import argparse
import os
import psutil
import time
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from util.mesh import Mesh

def compute_wa(mesh):
    edge_faces = {}
    for face in mesh.faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            if edge not in edge_faces:
                edge_faces[edge] = []
            edge_faces[edge].append(face)

    wa = sum(1 for edge, faces in edge_faces.items() if len(faces) != 2)

    # Print edges and associated faces
   
        
    return wa / len(edge_faces)
   




def compute_cd_ne(mesh1, mesh2):
    print(mesh1.vs.shape, mesh2.vs.shape)
    tree1 = KDTree(mesh1.vs)
    tree2 = KDTree(mesh2.vs)
    sampled_points = mesh2.vs[np.random.choice(len(mesh2.vs), size=50000)]
    d1, _ = tree1.query(sampled_points)
    d2, _ = tree2.query(sampled_points)
    cd = np.mean(d1) + np.mean(d2)
    
    # Sample corresponding points from mesh1 and mesh2
    sampled_points1 = mesh1.vs[np.random.choice(len(mesh1.vs), size=50000)]
    sampled_points2 = mesh2.vs[np.random.choice(len(mesh2.vs), size=50000)]
    
    # Compute the normal dissimilarity (NE) between the sampled points
    ne = np.mean(np.abs(sampled_points1 - sampled_points2))
    
    return cd, ne

def compute_le(mesh1, mesh2):
    max_vertices = max(len(mesh1.vs), len(mesh2.vs))
    L1 = mesh1.laplacian(max_vertices)
    L2 = mesh2.laplacian(max_vertices)
    _, v1 = eigs(L1, k=200)
    _, v2 = eigs(L2, k=200)
    le = np.mean((v1 - v2)**2)
    return le

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
    
    wa = compute_wa(simp_mesh)
    cd, ne = compute_cd_ne(mesh, simp_mesh)  # This uses the vertex normals
    le = compute_le(mesh, simp_mesh)
    print(f"Non-watertight edges (WA): {wa}")
    print(f"Chamfer distance (CD): {cd}")
    print(f"Normal dissimilarity (NE): {ne}")
    print(f"MSE of Laplacian eigenvectors (LE): {le}")

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
