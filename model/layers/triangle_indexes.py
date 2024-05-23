import torch
import torch.nn as nn

class TriangleIndexes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adjacency_matrix):
        # tensor of indexes of each neighbors of each nodes 
        nonzero = adjacency_matrix.nonzero()
        neighbors_one_indexes = nonzero.reshape(adjacency_matrix.shape[0],15,2)[:,:,1].clone()
        neighbors_two_indexes = neighbors_one_indexes[neighbors_one_indexes]        # Tensor for each 2 neighbors for each nodes (neighbors of neighbors)
        neighbors_three_indexes = neighbors_one_indexes[neighbors_two_indexes]      # Tensor for each 3 neighbors for each nodes (neighbors of neighbors of neighbors)

        # Find the indices where the current index is present along the last dimension => where start node = final node (= cycle)
        values_index_reshape = torch.arange(neighbors_three_indexes.shape[0]).repeat((15,15,15,1)).T
        indices = (neighbors_three_indexes == values_index_reshape).nonzero()

        i, j, k, l = indices[:,0], indices[:,1], indices[:,2], indices[:,3]         # First node index, Second node index, third node index, Fourth node index
        temp_j = neighbors_one_indexes[i,j]                                         # number of the nodes firsts neighbors
        temp_k = neighbors_two_indexes[i,j,k]                                       # number of the nodes seconds neighbors
        temp_l = neighbors_three_indexes[i,j,k,l]                                   # number of the nodes thirds neighbors
        triangles_indexes_test = torch.stack((i, temp_j, temp_k, temp_l), dim=1)    # nodes for each path 
        triangles_indexes_test = triangles_indexes_test[:,:3]                       # remove virtual 4th point (same as the first one (cycle))


        # filter triangles indexes to clean the clones (=> divide the number of triangles by 6)
        sorted_tensor, _ = torch.sort(triangles_indexes_test, dim=-1)
        triangles_ids_igraph = torch.unique(sorted_tensor, dim=0)

        return triangles_ids_igraph