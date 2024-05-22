import torch
import torch.nn as nn
import numpy as np

class RMatrix(nn.Module):
    def __init__(self, triangles, barycenters, indices_neigh_tri, number_neigh_tri):
        super().__init__()
        self.triangles = triangles
        self.barycenters = barycenters
        self.indices_neigh_tri = indices_neigh_tri
        self.number_neigh_tri = number_neigh_tri

    def forward(self):
        # DIFF BARYCENTERS
        barycenters_diff = np.subtract(self.barycenters[self.indices_neigh_tri[:, 0]][:, np.newaxis], self.barycenters[self.indices_neigh_tri[:, 1:]])   #Inverser la différence des barycentres si nécéssaire


        # TRIANGLE EDGES NORM
        v0, v1, v2 = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]

        # Calculate edge vectors
        e_ij = torch.norm(v0 - v1, dim=1)
        e_ik = torch.norm(v0 - v2, dim=1)
        e_jk = torch.norm(v1 - v2, dim=1)

        # Stack the edge vectors along the last dimension
        diff_vectors = torch.stack([e_ij, e_ik, e_jk], dim=1)


        # MAX/MIN DIFF VECTORS
        max_diff_vectors = diff_vectors.max(dim=1).values       # calculate t_n_max
        min_diff_vectors = diff_vectors.min(dim=1).values       # calculate t_n_min

        max_diff_vectors_diff = max_diff_vectors[self.indices_neigh_tri[:, 0]][:, None] - max_diff_vectors[self.indices_neigh_tri[:, 1:]]   #Inverser la différence des barycentres si nécéssaire   # calculate t_n_max - t_m_max
        min_diff_vectors_diff = min_diff_vectors[self.indices_neigh_tri[:, 0]][:, None] - min_diff_vectors[self.indices_neigh_tri[:, 1:]]   #Inverser la différence des barycentres si nécéssaire   # calculate t_n_min - t_m_min


        # R MATRIX COMPUTATION
        r_matrix = torch.zeros((self.triangles.shape[0], self.number_neigh_tri-1, 5))

        r_matrix[:, :, 0]   = min_diff_vectors_diff
        r_matrix[:, :, 1]   = max_diff_vectors_diff
        r_matrix[:, :, 2:5] = barycenters_diff
        
        return r_matrix