import torch
import torch.nn as nn
import time

import torchviz

from model.model_point_picker import ModelPointPicker
from model.layers.knn_simple import KNNSimple
from model.layers.devconv_gnn import DevConvGNN
from model.layers.sparse_attention_edge_predictor_layer import SparseAttentionEdgePredictorLayer
from model.layers.face_candidate import FaceCandidatesLayer
from model.layers.triangle_indexes import TriangleIndexes
from model.layers.first_p_init import FirstPInitLayer
from model.layers.knn import KNN
from model.layers.r_matrix import RMatrix
from model.layers.mlp import MLP



class ModelTriangleGenerator(nn.Module):
    def __init__(self, number_neigh_tri, debug=False):
        super().__init__()
        self.debug = debug
        
        self.number_neigh_tri = number_neigh_tri
        
        self.original_barycenters = None

        k = 20

        self.model_point_picker = ModelPointPicker()
        self.model_point_picker.load_state_dict(torch.load('save_models/point_picker/1/360.pt'))

        self.layer_knn_simple = KNNSimple(k=15)
        self.layer_devconv_edge_predictor = DevConvGNN(3, 64, 1)
        self.layer_sparse_attention_edge_predictor = SparseAttentionEdgePredictorLayer(64)
        self.layer_face_cadidates = FaceCandidatesLayer()
        self.layer_triangle_indexes = TriangleIndexes()
        self.layer_first_p_init = FirstPInitLayer()
        self.layer_knn = KNN(k=k, batch_size=1000)
        self.layer_r_matrix = RMatrix()
        self.layer_mlp = MLP(1, 128, 1)


    def forward(self, target_number_point, original_graph):
        # POINT SAMPLER
        with torch.no_grad():
            self.score_original_points, self.generated_graph_nodes = self.model_point_picker(target_number_point, original_graph)
        generated_graph_adjacency_matrix = self.layer_knn_simple(self.generated_graph_nodes)


        # EDGE PREDICTOR
        edge_index_generated_graph = generated_graph_adjacency_matrix.nonzero().T
        score_edge = self.layer_devconv_edge_predictor(edge_index_generated_graph, self.generated_graph_nodes).squeeze().sigmoid()
        S = self.layer_sparse_attention_edge_predictor(score_edge, generated_graph_adjacency_matrix)


        # FACE CANDIDATES
        generated_probabilistic_graph_adjacency_matrix = self.layer_face_cadidates(S, generated_graph_adjacency_matrix)

        # FACE CLASSIFIER
        triangles_ids_igraph = self.layer_triangle_indexes(generated_graph_adjacency_matrix)
        triangles = self.generated_graph_nodes[triangles_ids_igraph]

        barycenters = triangles.mean(dim=1)
        self.original_barycenters = barycenters

        indices_neigh_tri = self.layer_knn(barycenters).int()  #change datatype

        r_matrix = self.layer_r_matrix(triangles, barycenters, indices_neigh_tri, self.number_neigh_tri)

        p_init = self.layer_first_p_init(triangles_ids_igraph, generated_probabilistic_graph_adjacency_matrix, triangles)
        self.final_scores = self.layer_mlp(p_init, r_matrix, indices_neigh_tri)

        target_number_triangles = target_number_point // 3
        self.selected_triangles_probabilities, self.selected_triangles_indexes = torch.topk(self.final_scores, k=target_number_triangles)
        selected_triangles = triangles[self.selected_triangles_indexes]

        return selected_triangles