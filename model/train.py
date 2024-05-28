import torch
import numpy as np
import networkx as nx

from transformation import Transformation
from mesh_dataset import MeshDataset
from torch.utils.data import DataLoader
from layers.gnn_simplification_model import GNNSimplificationMesh
from loss.loss import total_loss




transformation = Transformation()

number_neigh_tri = 20
stl_file_path = "3d_models/stl/Handle.stl"
mesh_data = transformation.stl_to_mesh(stl_file_path)
graph = transformation.mesh_to_graph(mesh_data)


if len(graph._node)<20:
    raise Exception("Input mesh does not have enough vertices. (More than 20 is needed)")

graph_nodes = torch.Tensor(np.array(graph))
graph_adjacency_matrix = torch.Tensor(nx.adjacency_matrix(graph).toarray())


torch_dataset = MeshDataset("3d_models/stl/")
train_dataloader = DataLoader(torch_dataset, batch_size=1, shuffle=True)


gnn_model = GNNSimplificationMesh(number_neigh_tri)
# optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-5, weight_decay=0.99)

list(gnn_model.parameters())


for epoch in range(0, 5): 
    print(f'Starting epoch {epoch+1}')
    
    current_loss = 0.0
    


    # for data in train_dataloader:
        
    graph_nodes, graph_adjacency_matrix = torch_dataset[0]
        # optimizer.zero_grad()
    graph_nodes, graph_adjacency_matrix = torch.Tensor(graph_nodes), torch.Tensor(graph_adjacency_matrix)
    selected_triangles = gnn_model(200, graph_nodes, graph_adjacency_matrix)
        
        # loss = total_loss(outputs, targets)
        # loss.backward()
        # optimizer.step()
        
#         current_loss += loss.item()
#         if i % 500 == 499:
#             print('Loss after mini-batch %5d: %.3f' %
#                 (i + 1, current_loss / 500))
#             current_loss = 0.0
            
# print('Training process has finished.')