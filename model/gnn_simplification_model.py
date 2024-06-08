import torch
import torch.nn as nn
import time

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

        k = 20

        self.layer_gnn_model = GNN_Model()
        self.layer_multinomial = MultinomialLayer()
        self.layer_knn_simple = KNNSimple(k=15)
        self.layer_devconv_edge_predictor = DevConv(64)
        self.layer_sparse_attention_edge_predictor = SparseAttentionEdgePredictorLayer(64)
        self.layer_face_cadidates = FaceCandidatesLayer()
        self.layer_triangle_indexes = TriangleIndexes()
        self.layer_triangle_nodes = TriangleNodes()
        self.layer_first_p_init = FirstPInitLayer()
        self.layer_barycenters = BarycentersLayer()
        self.layer_knn = KNN(k=k, batch_size=100)
        self.layer_r_matrix = RMatrix()
        self.layer_mlp = MLP(128, k)


    def forward(self, user_number_triangles, graph_nodes, graph_adjacency_matrix):
        # POINT SAMPLER
        
        start = time.time()
        self.inclusion_score = self.layer_gnn_model(torch.empty(0), graph_nodes, graph_adjacency_matrix)
        end = time.time()
        print('gnn_model : ', end - start)

        start = time.time()
        target_number_point = min(graph_nodes.shape[0], user_number_triangles*3)   # number of points for the simplification
        self.extended_graph_nodes = self.layer_multinomial(self.inclusion_score, target_number_point, graph_nodes)
        end = time.time()
        print('multinomial : ', end - start)

        start = time.time()
        extended_graph_adjacency_matrix = self.layer_knn_simple(self.extended_graph_nodes)
        end = time.time()
        print('knn simple : ', end - start)


        # EDGE PREDICTOR
        start = time.time()
        inclusion_score_edge = self.layer_devconv_edge_predictor(torch.empty((0)), self.extended_graph_nodes,extended_graph_adjacency_matrix, return_flatten=False)
        end = time.time()
        print('simp^le devconv : ', end - start)

        start = time.time()
        f = torch.mean(inclusion_score_edge, dim=1)                            # Flatten the matrix of inclusion score
        S = self.layer_sparse_attention_edge_predictor(f, extended_graph_adjacency_matrix)
        end = time.time()
        print('sparse attention edge predictor : ', end - start)


        # FACE CANDIDATES
        start = time.time()
        A_s = self.layer_face_cadidates(torch.Tensor(S), extended_graph_adjacency_matrix)
        end = time.time()
        print('face candidate : ', end - start)


        # FACE CLASSIFIER
        start = time.time()
        triangles_ids_igraph = self.layer_triangle_indexes(extended_graph_adjacency_matrix)
        end = time.time()
        print('triangle indexes : ', end - start)

        start = time.time()
        triangles = self.layer_triangle_nodes(triangles_ids_igraph, self.extended_graph_nodes)
        end = time.time()
        print('traingle node : ', end - start)

        start = time.time()
        p_init = self.layer_first_p_init(triangles_ids_igraph, A_s, triangles)
        end = time.time()
        print('first p init : ', end - start)

        start = time.time()
        barycenters = self.layer_barycenters(triangles)
        end = time.time()
        print('barycenter : ', end - start)

        start = time.time()
        indices_neigh_tri = self.layer_knn(barycenters).int()  #change datatype
        end = time.time()
        print('knn : ', end - start)

        start = time.time()
        r_matrix = self.layer_r_matrix(triangles, barycenters, indices_neigh_tri, self.number_neigh_tri)
        end = time.time()
        print('r matrix : ', end - start)

        start = time.time()
        self.final_scores = self.layer_mlp(p_init, r_matrix, indices_neigh_tri)
        end = time.time()
        print('mlp : ', end - start)

        start = time.time()
        self.selected_triangles_indexes = torch.topk(self.final_scores, k=user_number_triangles).indices
        selected_triangles = triangles[self.selected_triangles_indexes]
        end = time.time()
        print('selected triangles topk : ', end - start)

        return selected_triangles