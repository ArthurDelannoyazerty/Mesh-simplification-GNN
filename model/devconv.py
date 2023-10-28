import torch
import torch.nn as nn

class DevConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DevConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inclusion_score = None

        # Learnable parameters
        self.W_phi = nn.Parameter(torch.rand(in_channels, out_channels))
        self.W_theta = nn.Parameter(torch.rand(in_channels, out_channels))

    def forward(self, x, adjacency_matrix, inclusion_score=None):
        # Ensure dimensions match
        assert x.size(1) == self.in_channels, "Input tensor dimension mismatch"
        assert adjacency_matrix.size(0) == adjacency_matrix.size(1) == x.size(0), "Adjacency matrix size mismatch"

        if inclusion_score==None:
            self.inclusion_score = torch.zeros(x.size(0))

        # Compute the deviation of each point from its neighbors
        deviations = torch.zeros_like(x)  # Initialize deviations as zeros

        for i in range(x.size(0)):
            neighbors = adjacency_matrix[i]  # Get neighbors of point i
            if neighbors.sum() == 0:
                continue  # Skip if there are no neighbors
            
            temp = [((x[i][0]-x[j][0]) + (x[i][1]-x[j][1]) + (x[i][2]-x[j][2])) if neighbors[j]!=0 else 0 for j in range(len(neighbors))]
            deviation = torch.mul(torch.FloatTensor(temp), self.W_phi)
            maxi = torch.max(deviation)
            deviations[i] = maxi

        return deviations

# Example usage:
# Create a random input tensor and an adjacency matrix
input_dim = 3  # Input tensor has 64 dimensions
num_points = 10
x = torch.rand(num_points, input_dim)
adjacency_matrix = torch.randint(2, (num_points, num_points))

# Make sure adjacency_matrix[i][i] is set to 0, as a point is not its own neighbor
for i in range(num_points):
    adjacency_matrix[i][i] = 0

# Create and apply the DevConv layer
devconv = DevConvLayer(input_dim, out_channels=10)
output = devconv(x, adjacency_matrix)
print(output)
