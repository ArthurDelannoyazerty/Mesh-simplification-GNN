import torch
import torch.nn as nn

class SparseAttentionEdgePredictorLayer(nn.Module):
    def __init__(self, nodes, neighbors, size=64):
        super().__init__()
        self.size = size
        self.nodes = nodes
        self.neighbors = neighbors
        self.wq = nn.Parameter(torch.Tensor(size))
        self.wk = nn.Parameter(torch.Tensor(size))

        nn.init.normal_(self.wq)
        nn.init.normal_(self.wk)

    def forward(self, f):
        wq_f = self.wq.reshape(-1, 1) * f                   # Wq*f
        wk_f = self.wk.reshape(-1, 1) * f                   # Wq*f
        S = torch.exp(torch.matmul(wq_f.T, wk_f))           # e^((wq_f.T)*(wk_f))
        
        nonzero_neigh = self.neighbors.nonzero()                                                    # Find indexes of neighbors in graph
        unique_first_elements, counts = torch.unique(nonzero_neigh[:, 0], return_counts=True)       # Count number of neighbors per node
        split_tensors = list(torch.split(nonzero_neigh, tuple(counts)))                             # split indexes of neighbors into a list (1 element = 1 tensor of indexes)

        temp = [[S[n[i,0], n[i,1]] for i in range(len(n))] for n in split_tensors]                  # For each node, get the S value for the neighbors indexes
        summed = torch.Tensor([torch.sum(torch.Tensor(e)) for e in temp])                           # Sum these results for each nodes
        division = summed.unsqueeze(0).repeat(1, S.shape[1], 1)[0]                                  # Repeat the sum in S.shape[1] array => division per columns
        final_term  = S / division

        return final_term