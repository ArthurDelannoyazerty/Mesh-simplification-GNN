import torch
import torch.nn as nn

from layers.gnn import GNN_Model
from layers.multinomial import MultinomialLayer
from layers.knn_simple import KNNSimple
from layers.devcon import DevConv
from layers.sparse_attention_edge_predictor_layer import SparseAttentionEdgePredictorLayer
from layers.face_candidate import FaceCandidatesLayer
from layers.triangle_indexes import TriangleIndexes
from layers.triangle_nodes import TriangleNodes
from layers.first_p_init import FirstPInitLayer
from layers.barycenter import BarycentersLayer
from layers.knn import KNN
from layers.r_matrix import RMatrix
from layers.mlp import MLP

class GNNSimplificationMesh(nn.Module):
    def __init__(self, number_neigh_tri):
        super().__init__()
        # self.graph_nodes = graph_nodes
        # self.graph_adjacency_matrix = graph_adjacency_matrix
        self.number_neigh_tri = number_neigh_tri

    def forward(self, user_number_triangles, graph_nodes, graph_adjacency_matrix):
        # POINT SAMPLER
        gnn = GNN_Model(graph_nodes, graph_adjacency_matrix)
        inclusion_score = gnn(torch.empty(0))

        target_number_point = min(graph_nodes.shape[0], user_number_triangles*3)   # number of points for the simplification
        layer = MultinomialLayer(target_number_point, graph_nodes)
        extended_graph_nodes = layer.forward(inclusion_score)

        extended_graph_adjacency_matrix = KNNSimple(k=15)(extended_graph_nodes)

        # EDGE PREDICTOR
        devconv = DevConv(extended_graph_nodes,extended_graph_adjacency_matrix, 64)
        inclusion_score_edge = devconv(previous_inclusion_score=torch.empty((0)), return_flatten=False)

        f = torch.mean(inclusion_score_edge, dim=1)                            # Flatten the matrix of inclusion score
        layer = SparseAttentionEdgePredictorLayer(extended_graph_nodes, extended_graph_adjacency_matrix)
        S = layer.forward(f)

        # FACE CANDIDATES
        layer = FaceCandidatesLayer(extended_graph_adjacency_matrix)
        A_s = layer(torch.Tensor(S))

        # FACE CLASSIFIER
        layer_find_triangles_indexes = TriangleIndexes(extended_graph_adjacency_matrix)
        triangles_ids_igraph = layer_find_triangles_indexes()

        layer_get_triangles = TriangleNodes(extended_graph_nodes)
        triangles = layer_get_triangles(triangles_ids_igraph)

        p_init_layer = FirstPInitLayer(A_s, triangles)
        p_init = p_init_layer(triangles_ids_igraph)

        barycenters_layer = BarycentersLayer()
        barycenters = barycenters_layer(triangles)

        knn_layer = KNN(barycenters)
        indices_neigh_tri = knn_layer(barycenters).int()  #change datatype

        r_matrix_layer = RMatrix(triangles, barycenters, indices_neigh_tri, self.number_neigh_tri)
        r_matrix = r_matrix_layer()

        mlp = MLP(r_matrix, indices_neigh_tri, 128)
        final_scores = mlp(p_init)

        selected_triangles_indexes = torch.topk(final_scores, k=user_number_triangles).indices
        selected_triangles = triangles[selected_triangles_indexes]

        return selected_triangles