import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_file import read_stl_file, read_obj_file

def viewsimplify(original_file, vertices_file, faces_file, fraction):
    # Read original mesh
    original_vertices, original_faces = read_stl_file(original_file)

    # Read simplified vertices and faces
    simplified_vertices = np.loadtxt(vertices_file)
    simplified_faces = np.loadtxt(faces_file, dtype=int)

    # Create 3D plots for original and simplified meshes
    fig = plt.figure(figsize=(12, 6))
    
    # Plot original mesh
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Original Mesh')
    ax1.plot_trisurf(original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2], triangles=original_faces, color='b')

    # Plot simplified mesh
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f'Simplified Mesh ({fraction} Fraction)')
    ax2.plot_trisurf(simplified_vertices[:, 0], simplified_vertices[:, 1], simplified_vertices[:, 2], triangles=simplified_faces, color='r')

    plt.show()

# Example usage:
original_file = 'ankylosaurus-ascii.stl'
vertices_file = 'ankylosaurus-ascii_vertices_0.5.txt'
faces_file = 'ankylosaurus-ascii_faces_0.5.txt'
fraction = 0.5

viewsimplify(original_file, vertices_file, faces_file, fraction)
