import networkx as nx
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def stl_to_mesh(stl_file_path):
    """Load a stl file and transform it into a mesh. Return the mesh."""
    return mesh.Mesh.from_file(stl_file_path)

def mesh_to_graph(mesh_data):
    """Transform a mesh in a graph. Return the graph."""
    G = nx.Graph()

    for triangle in mesh_data.vectors:
        for vertex in triangle:
            vertex_tuple = tuple(vertex)
            G.add_node(vertex_tuple)

        # Add edges between vertices of the same triangle
        for j in range(3):
            edge = tuple(triangle[j]), tuple(triangle[(j + 1) % 3])
            G.add_edge(*edge)
    return G

def mesh_to_display(mesh_data, display_faces=True, display_vertices=True):
    """Plot the mesh in a 3D model."""
    figure = plt.figure("Mesh")
    axes = figure.add_subplot(projection='3d')

    # Load the STL files and add the vectors to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh_data.vectors, edgecolor=('r' if display_vertices else None), alpha=(0.35 if display_faces else 0.0)))
    scale = mesh_data.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    plt.show()

def graph_to_display(graph, mesh_data, display_faces=True, display_vertices=True, display_nodes=False):
    """Display the vertices and nodes from graph, and face from mesh. Watch to nodes and vertices if something strange !"""
    fig = plt.figure("Graph")
    ax = fig.add_subplot(111, projection='3d')

    # Faces
    if display_faces:
        poly3d = mplot3d.art3d.Poly3DCollection(mesh_data.vectors, linewidths=0, alpha=0.35)
        ax.add_collection3d(poly3d)

    # Nodes
    if display_nodes:
        nodes = list(graph.nodes())
        x, y, z = zip(*nodes)
        ax.scatter(x, y, z, c='b', marker='o')

    # Edges
    if display_vertices:
        for edge in list(graph.edges()):
            u, v = edge
            ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], c='r')

    scale = mesh_data.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    # Set labels and display the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def print_graph_properties(graph, display_graph=False):
    print(f"Number of nodes: {len(graph.nodes())}")
    print(f"Number of edges: {len(graph.edges())}")
    if display_graph: nx.draw(graph)

# Create objects
stl_file_path = "3d_models/cube.stl"
mesh_data = stl_to_mesh(stl_file_path)
graph = mesh_to_graph(mesh_data)

# Display informations
print_graph_properties(graph, display_graph=True)
mesh_to_display(mesh_data, display_faces=True, display_vertices=True)
graph_to_display(graph, mesh_data, display_faces=True, display_vertices=True, display_nodes=True)