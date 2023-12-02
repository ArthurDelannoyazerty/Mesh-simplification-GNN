import networkx as nx
from stl import mesh
import vtkplotlib as vpl
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import pyvista as pv
from vtkplotlib import geometry

class Transformation():
    def __init__(self, debug=False):
        self.DEBUG = debug

    def stl_to_mesh(self, stl_file_path):
        """Load a stl file and transform it into a mesh. Return the mesh."""
        if self.DEBUG: print("DEBUG : Load mesh from file")
        return mesh.Mesh.from_file(stl_file_path)

    def mesh_to_graph(self, mesh_data):
        """Transform a mesh in a graph. Return the graph."""
        G = nx.Graph()
        nb_triangle = len(mesh_data.vectors)

        for index_triangle, triangle in enumerate(mesh_data.vectors):
            if self.DEBUG: print("DEBUG : Mesh to graph : Triangle " + str(index_triangle+1) + "/" + str(nb_triangle))
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

    def mesh_to_display_vtk(self, mesh):
        if self.DEBUG: print("DEBUG : Display mesh vtk")
        # tri_scalars = np.inner(mesh.units, np.array([0, 0, 1]))
        vertices = mesh.vectors
        vpl.plot(vertices, join_ends=True, color="dark red")
        vpl.mesh_plot(mesh)
        # vpl.mesh_plot_with_edge_scalars(mesh)
        vpl.show()

    def print_graph_properties(self, graph, display_graph=False, display_labels=False):
        if self.DEBUG: print("DEBUG : Display graph properties")
        print(f"Number of nodes: {len(graph.nodes())}")
        print(f"Number of edges: {len(graph.edges())}")
        if display_graph: 
            if self.DEBUG: print("DEBUG : display graph")
            nx.draw(graph, with_labels=display_labels)
            plt.show()

    def graph_to_mesh(self, graph):
        vertices = list()
        dict_face = dict()  #{face1: {index1, index2}, ...}
        index_vertices = 0
        len_vertices = len(graph._node.items())
        for coord, set_index_triangle in graph._node.items():
            if self.DEBUG: print(f"DEBUG : Graph to mesh :  Step 1 : {index_vertices+1}/{len_vertices}")
            vertices.append(list(coord))
            index_triangle = set_index_triangle["index_triangle"]
            for index in index_triangle:
                if index not in dict_face:
                    dict_face[index] = set()
                dict_face[index].add(index_vertices)
            index_vertices += 1

        faces = [list(triple_index) for triple_index in dict_face.values()]

        faces, vertices = np.array(faces), np.array(vertices)

        mesh_obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        len_faces = len(faces)
        for i, f in enumerate(faces):
            if self.DEBUG: print("DEBUG : Graph to mesh : Step 2 : " + str(i+1) + "/" + str(len_faces))
            for j in range(3):
                mesh_obj.vectors[i][j] = vertices[f[j],:]
        return mesh_obj



# transformation = Transformation(True)

# # Create objects
# stl_file_path = "3d_models/stl/Handle.stl"
# mesh_data = transformation.stl_to_mesh(stl_file_path)
# graph = transformation.mesh_to_graph(mesh_data)
# mesh_from_graph = transformation.graph_to_mesh(graph)

# # Display informations
# transformation.print_graph_properties(graph, display_graph=False, display_labels=True)
# transformation.mesh_to_display_vtk(mesh_data)
# transformation.mesh_to_display_vtk(mesh_from_graph)