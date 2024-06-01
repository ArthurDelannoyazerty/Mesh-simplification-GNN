import os
import networkx as nx
import numpy as np

import torch
from torch.utils.data import Dataset
from transformation import Transformation

class MeshDataset(Dataset):
    def __init__(self, mesh_dir):
        self.mesh_dir = mesh_dir
        self.filepaths = list()
        for filename in os.listdir(mesh_dir):
            f = os.path.join(mesh_dir, filename)
            if os.path.isfile(f):
                self.filepaths.append(f)
        self.len = len(self.filepaths)
        self.transformation = Transformation()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mesh_path = self.filepaths[idx]
        mesh_data = self.transformation.stl_to_mesh(mesh_path)
        graph = self.transformation.mesh_to_graph(mesh_data)

        adjacency_coo = nx.adjacency_matrix(graph).tocoo()
        indices = torch.stack((torch.tensor(adjacency_coo.row, dtype=torch.long), 
                               torch.tensor(adjacency_coo.col, dtype=torch.long)))
        values = torch.tensor(adjacency_coo.data, dtype=torch.float32)
        adjacency_sparse = torch.sparse_coo_tensor(indices, values, torch.Size(adjacency_coo.shape))
        
        graph_nodes = torch.Tensor(np.array(graph.nodes))
        
        return graph_nodes, adjacency_sparse