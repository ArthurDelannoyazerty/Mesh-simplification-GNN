import networkx as nx
from stl import mesh
import vtkplotlib as vpl
import numpy as np

def stl_to_mesh(stl_file_path):
    """Load a stl file and transform it into a mesh. Return the mesh."""
    return mesh.Mesh.from_file(stl_file_path)

def mesh_to_graph(mesh_data):
    """Transform a mesh in a graph. Return the graph."""
    G = nx.Graph()

    for index_triangle, triangle in enumerate(mesh_data.vectors):
        # for vertex in triangle:
        #     vertex_tuple = tuple(np.hstack((vertex, index_triangle)))
        #     G.add_node(vertex_tuple)

        # Add edges between vertices of the same triangle
        for j in range(3):
            edge = tuple(np.hstack((triangle[j], index_triangle))), tuple(np.hstack((triangle[(j + 1) % 3], index_triangle)))
            G.add_edge(*edge)
    return G

def mesh_to_display_vtk(mesh):
    tri_scalars = np.inner(mesh.units, np.array([0, 0, 1]))
    vpl.mesh_plot(mesh, tri_scalars=tri_scalars)
    # vpl.mesh_plot_with_edge_scalars(mesh)
    vpl.show()

def print_graph_properties(graph, display_graph=False):
    print(f"Number of nodes: {len(graph.nodes())}")
    print(f"Number of edges: {len(graph.edges())}")
    if display_graph: nx.draw(graph)

def graph_to_mesh(graph):
    # creation node array
    x, y, z, triangle_id_list = zip(*graph.nodes())
    vertices = np.array(list(zip(x,y,z)))
    
    # creation vertice array
    nb_triangle = int(max(triangle_id_list))
    faces = [[] for _ in range(nb_triangle+1)]
    for i, element in enumerate(triangle_id_list):
        faces[int(element)].append(i)
    faces = np.array(faces)

    # Create the mesh object
    mesh_obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_obj.vectors[i][j] = vertices[f[j],:]
    if not mesh_obj.check():
        raise BaseException("Graph to mesh error : mesh not valid")
    return mesh_obj


# Create objects
stl_file_path = "3d_models/Handle.stl"
mesh_data = stl_to_mesh(stl_file_path)
graph = mesh_to_graph(mesh_data)
mesh_from_graph = graph_to_mesh(graph)

# Display informations
mesh_to_display_vtk(mesh_data)
mesh_to_display_vtk(mesh_from_graph)
print_graph_properties(graph, display_graph=True)