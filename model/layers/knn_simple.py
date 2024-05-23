import torch
import torch.nn as nn

class KNNSimple(nn.Module):
    """
    Create a graph based on k-nearest neighbors using PyTorch.
    Parameters:
    - nodes: Tensor of shape (n, 3) representing 3D nodes.
    - k: Number of nearest neighbors.
    Returns:
    - adjacency_matrix: Binary adjacency matrix representing the graph.
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, nodes):
        expanded_x1 = nodes.unsqueeze(1)
        expanded_x2 = nodes.unsqueeze(0)
        distances = torch.norm(expanded_x1 - expanded_x2, dim=2)        # distance matrix

        _, indices = torch.topk(distances, self.k + 1, largest=False, sorted=True, dim=1)
        indices = indices[:, 1:]  # Exclude the node itself

        # Create adjacency matrix
        adjacency_matrix = torch.zeros(nodes.shape[0], nodes.shape[0], dtype=torch.float32)
        adjacency_matrix.scatter_(1, indices, 1)

        return adjacency_matrix