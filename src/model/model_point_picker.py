import torch
import time
from model.layers.gnn import GNN_Model
from model.layers.multinomial import MultinomialLayer


class ModelPointPicker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_gnn_model = GNN_Model()
        self.layer_multinomial = MultinomialLayer()


    def forward(self, target_number_point, original_graph):
        score_original_points = self.layer_gnn_model(original_graph).squeeze()
        
        generated_graph_nodes = self.layer_multinomial(score_original_points, target_number_point, original_graph.x)

        return score_original_points, generated_graph_nodes
