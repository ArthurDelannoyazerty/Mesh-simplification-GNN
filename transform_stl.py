import networkx as nx
from stl import mesh
import vtkplotlib as vpl
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

def stl_to_mesh(stl_file_path):
    """Load a stl file and transform it into a mesh. Return the mesh."""
    return mesh.Mesh.from_file(stl_file_path)

def mesh_to_graph(mesh_data):
    """Transform a mesh in a graph. Return the graph."""
    G = nx.Graph()

    for index_triangle, triangle in enumerate(mesh_data.vectors):
        # for vertex in triangle:
        #     vertex_tuple = tuple(vertex)
        #     G.add_node(vertex_tuple)

        # Add edges between vertices of the same triangle
        for j in range(3):
            current_node = tuple(triangle[j])
            edge = current_node, tuple(triangle[(j + 1) % 3])
            G.add_edge(*edge)

            # if attribute do not exists
            if len(G.nodes[current_node])==0:
                G.nodes[current_node]['index_triangle'] = set()
            G.nodes[current_node]['index_triangle'].add(index_triangle)
    return G

def mesh_to_display_vtk(mesh):
    # tri_scalars = np.inner(mesh.units, np.array([0, 0, 1]))
    vpl.mesh_plot(mesh)
    # vpl.mesh_plot_with_edge_scalars(mesh)
    vpl.show()

def print_graph_properties(graph, display_graph=False, display_labels=False):
    print(f"Number of nodes: {len(graph.nodes())}")
    print(f"Number of edges: {len(graph.edges())}")
    if display_graph: 
        nx.draw(graph, with_labels=display_labels)
        plt.show()

def graph_to_mesh(graph):
    # Create a dictionary to store the integers as keys and their corresponding coordinates as values
    integer_to_coordinates = {}
    set_coordinate = set()

    # Iterate through the dictionary and populate the integer_to_coordinates dictionary
    for coord, attributes in graph._node.items():
        set_coordinate.add(coord)
        index_set = attributes['index_triangle']
        for index in index_set:
            if index in integer_to_coordinates:
                integer_to_coordinates[index].append(coord)
            else:
                integer_to_coordinates[index] = [coord]

    vertices = [list(coord) for coord in set_coordinate]

    # Create a list to store the triplets with the same integer
    faces = [[] for _ in range(int(max(integer_to_coordinates.keys()))+1)]

    for i, coord in enumerate(vertices):
        for j, triplet in integer_to_coordinates.items():
            if tuple(coord) in triplet:
                faces[j].append(i)

    faces = np.array(faces)
    vertices = np.array(vertices)

    mesh_obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_obj.vectors[i][j] = vertices[f[j],:]
    return mesh_obj


# Create objects
stl_file_path = "3d_models/cube.stl"
mesh_data = stl_to_mesh(stl_file_path)
graph = mesh_to_graph(mesh_data)
mesh_from_graph = graph_to_mesh(graph)

# Display informations
print_graph_properties(graph, display_graph=True, display_labels=True)
mesh_to_display_vtk(mesh_data)
mesh_to_display_vtk(mesh_from_graph)
