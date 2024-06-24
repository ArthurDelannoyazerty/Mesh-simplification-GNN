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

        # Create COO indices and values for the adjacency matrix
        coo_indices = torch.cat([edges, edges.flip(1)], dim=0).t()
        values = torch.ones(coo_indices.size(1), dtype=torch.float)

        # Convert COO indices to CSR format
        row_indices = coo_indices[0]
        col_indices = coo_indices[1]

        # Calculate the number of non-zero elements per row
        row_counts = torch.bincount(row_indices, minlength=num_nodes)

        # Calculate the cumulative sum of row_counts to get the CSR row pointers
        csr_row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(row_counts, dim=0)])

        # Create the CSR tensor
        adjacency_matrix_csr = torch.sparse_csr_tensor(
            csr_row_ptr,
            col_indices,
            values,
            size=(num_nodes, num_nodes)
        )
        
        return vertices, adjacency_matrix_csr, triangles