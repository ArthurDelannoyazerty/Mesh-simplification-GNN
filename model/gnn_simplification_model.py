import torch
import torch.nn as nn
import time

import torchviz

from layers.gnn import GNN_Model
from layers.multinomial import MultinomialLayer
from layers.knn_simple import KNNSimple
from layers.devconv_gnn import DevConvGNN
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
    def __init__(self, number_neigh_tri, debug=False):
        super().__init__()
        self.debug = debug
        
        self.number_neigh_tri = number_neigh_tri
        
        self.original_barycenters = None

        k = 20

        self.layer_gnn_model = GNN_Model()
        self.layer_multinomial = MultinomialLayer()
        self.layer_knn_simple = KNNSimple(k=15)
        self.layer_devconv_edge_predictor = DevConvGNN(3, 64, 1)
        self.layer_sparse_attention_edge_predictor = SparseAttentionEdgePredictorLayer(64)
        self.layer_face_cadidates = FaceCandidatesLayer()
        self.layer_triangle_indexes = TriangleIndexes()
        self.layer_triangle_nodes = TriangleNodes()
        self.layer_first_p_init = FirstPInitLayer()
        self.layer_barycenters = BarycentersLayer()
        self.layer_knn = KNN(k=k, batch_size=1000)
        self.layer_r_matrix = RMatrix()
        self.layer_mlp = MLP(128, k)


    def forward(self, target_number_triangles, original_graph):
        # POINT SAMPLER
        
        start = time.time()
        self.score_original_points = self.layer_gnn_model(original_graph).squeeze()
        end = time.time()
        print('gnn_model : ', end - start)
        if self.debug: display(torchviz.make_dot(self.score_original_points))


        start = time.time()
        target_number_point = min(original_graph.x.shape[0], target_number_triangles*3)   # number of points for the simplification
        self.generated_graph_nodes = self.layer_multinomial(self.score_original_points, target_number_point, original_graph.x)
        end = time.time()
        print('multinomial : ', end - start)
        if self.debug: display(torchviz.make_dot(self.generated_graph_nodes))

        start = time.time()
        generated_graph_adjacency_matrix = self.layer_knn_simple(self.generated_graph_nodes)
        end = time.time()
        print('knn simple : ', end - start)
        if self.debug: display(torchviz.make_dot(generated_graph_adjacency_matrix))


        # EDGE PREDICTOR
        start = time.time()
        edge_index_generated_graph = generated_graph_adjacency_matrix.nonzero().T
        score_edge = self.layer_devconv_edge_predictor(edge_index_generated_graph, self.generated_graph_nodes).squeeze()
        end = time.time()
        print('simple devconv : ', end - start)
        if self.debug: display(torchviz.make_dot(score_edge))

        start = time.time()
        # f = torch.mean(score_edge, dim=1).sigmoid()                            # Flatten the matrix of inclusion score
        S = self.layer_sparse_attention_edge_predictor(score_edge, generated_graph_adjacency_matrix)
        end = time.time()
        print('sparse attention edge predictor : ', end - start)
        if self.debug: display(torchviz.make_dot(S))


        # FACE CANDIDATES
        start = time.time()
        generated_probabilistic_graph_adjacency_matrix = self.layer_face_cadidates(torch.Tensor(S), generated_graph_adjacency_matrix)
        end = time.time()
        print('face candidate : ', end - start)
        if self.debug: display(torchviz.make_dot(generated_probabilistic_graph_adjacency_matrix))


        # FACE CLASSIFIER
        start = time.time()
        triangles_ids_igraph = self.layer_triangle_indexes(generated_graph_adjacency_matrix)
        end = time.time()
        print('triangle indexes : ', end - start)
        if self.debug: display(torchviz.make_dot(triangles_ids_igraph))

        start = time.time()
        triangles = self.layer_triangle_nodes(triangles_ids_igraph, self.generated_graph_nodes)
        end = time.time()
        print('traingle node : ', end - start)
        if self.debug: display(torchviz.make_dot(triangles))

        start = time.time()
        p_init = self.layer_first_p_init(triangles_ids_igraph, generated_probabilistic_graph_adjacency_matrix, triangles)
        end = time.time()
        print('first p init : ', end - start)
        if self.debug: display(torchviz.make_dot(p_init))

        start = time.time()
        barycenters = self.layer_barycenters(triangles)
        self.original_barycenters = barycenters
        end = time.time()
        print('barycenter : ', end - start)
        if self.debug: display(torchviz.make_dot(self.original_barycenters))

        start = time.time()
        indices_neigh_tri = self.layer_knn(barycenters).int()  #change datatype
        end = time.time()
        print('knn : ', end - start)
        if self.debug: display(torchviz.make_dot(indices_neigh_tri))

        start = time.time()
        r_matrix = self.layer_r_matrix(triangles, barycenters, indices_neigh_tri, self.number_neigh_tri)
        end = time.time()
        print('r matrix : ', end - start)
        if self.debug: display(torchviz.make_dot(r_matrix))

        start = time.time()
        self.final_scores = self.layer_mlp(p_init, r_matrix, indices_neigh_tri)
        end = time.time()
        print('mlp : ', end - start)
        if self.debug: display(torchviz.make_dot(self.final_scores))

        start = time.time()
        self.selected_triangles_probabilities, self.selected_triangles_indexes = torch.topk(self.final_scores, k=target_number_triangles)
        selected_triangles = triangles[self.selected_triangles_indexes]
        end = time.time()
        print('selected triangles topk : ', end - start)
        if self.debug: display(torchviz.make_dot(self.selected_triangles_probabilities))
        if self.debug: display(torchviz.make_dot(selected_triangles))

        return selected_triangles