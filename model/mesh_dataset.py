import os
import networkx as nx
import numpy as np
import stl_reader
import pyvista

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

        vertices, triangles = stl_reader.read(mesh_path)
        vertices, triangles = torch.from_numpy(vertices), torch.from_numpy(triangles.astype(np.int32))
        num_nodes = vertices.size(0)
        
        edges = torch.cat([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], dim=0)

        adjacency_matrix = torch.sparse_coo_tensor(
            indices=torch.cat([edges, edges.flip(1)], dim=0).t(),
            values=torch.ones(edges.size(0) * 2, dtype=torch.float),
            size=(num_nodes, num_nodes)
        )
        
        return vertices, adjacency_matrix