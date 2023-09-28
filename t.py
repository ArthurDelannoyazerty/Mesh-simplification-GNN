import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
import networkx as nx

def plot_3d_model_with_faces(stl_file_path, graph):
    # Load the STL files and extract the faces
    your_mesh = mesh.Mesh.from_file(stl_file_path)
    faces = [face for face in your_mesh.vectors]
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection for the faces from the STL mesh
    poly3d = Poly3DCollection(faces, facecolors='cyan', linewidths=0, edgecolors='r', alpha=0.35)
    ax.add_collection3d(poly3d)
    # Extract node coordinates from the graph
    nodes = list(graph.nodes())
    x, y, z = zip(*nodes)

    # Extract edges as pairs of nodes
    edges = list(graph.edges())

    # Plot nodes
    ax.scatter(x, y, z, c='b', marker='o')

    # Plot edges
    for edge in edges:
        u, v = edge
        ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], c='r')



    # Set labels and display the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def stl_to_graph(stl_file_path):
    """Load the stl file and transform it into a graph."""
    mesh_data = mesh.Mesh.from_file(stl_file_path)
    G = nx.Graph()

    # Iterate through each triangle in the mesh
    for i, triangle in enumerate(mesh_data.vectors):
        # Add vertices to the graph
        for vertex in triangle:
            vertex_tuple = tuple(vertex)
            G.add_node(vertex_tuple)

        # Add edges between vertices of the same triangle
        for j in range(3):
            edge = tuple(triangle[j]), tuple(triangle[(j + 1) % 3])
            G.add_edge(*edge)
    return G

# Example usage:
# Assuming you have a graph named 'graph' from NetworkX and an STL file named 'your_stl_file.stl'
stl_file_path = "3d_models/cube.stl"  # Replace with the path to your STL file
graph = stl_to_graph(stl_file_path)
plot_3d_model_with_faces(stl_file_path, graph)