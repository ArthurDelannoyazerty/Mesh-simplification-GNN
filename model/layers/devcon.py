import torch
import torch.nn as nn

class DevConv(nn.Module):
    def __init__(self, output_dimension):
        super().__init__()
        self.size = output_dimension

        self.W_phi = nn.Parameter(torch.Tensor(output_dimension))
        self.W_theta = nn.Parameter(torch.Tensor(size=(3,1)))
        nn.init.normal_(self.W_phi)
        nn.init.normal_(self.W_theta)
    
    
    def forward(self, previous_inclusion_score, nodes, adjacency_matrix, return_flatten=True):
        
        if adjacency_matrix.layout==torch.sparse_coo:
            indices = adjacency_matrix.coalesce().indices()
            row_indices = indices[0]
            col_indices = indices[1]

            neighbors = nodes[col_indices]

            current_nodes = nodes[row_indices]
            diff = current_nodes - neighbors

            W_theta_diff = torch.matmul(diff, self.W_theta)

            neigh_distances = torch.norm(W_theta_diff, dim=1)

            max_distances = torch.zeros(nodes.shape[0], device=nodes.device)
            # max_distances.index_add_(0, row_indices, neigh_distances)

            unique_row_indices = torch.unique(row_indices)
            for i in unique_row_indices:
                max_distances[i] = neigh_distances[row_indices == i].max()
            list_inc_score = self.W_phi * max_distances.unsqueeze(1)
            # print(list_inc_score)

            unique_count_length = row_indices.bincount()
            

        elif adjacency_matrix.layout==torch.strided:
            list_inc_score = torch.zeros((nodes.shape[0], self.size), device="cuda" if torch.cuda.is_available() else "cpu")                                               #list of "output_dimension" for each "list_node" element
            for index_current_node, list_neighbors in enumerate(adjacency_matrix):                                  # for each node and its adjacency nodes
                neighbors = nodes[list_neighbors.nonzero()]                                                    # get neighbors nodes
                diff = nodes[index_current_node] - neighbors                                                   # Compute the differences between current_node and all neighbor nodes   (x_i - x_j)
                to_norm = self.W_theta.T.unsqueeze(1).repeat(1, diff.shape[0], 1)[0] * diff.squeeze(1)              # Compute W_theta * (x_i - x_j)
                neigh_distances = torch.norm(to_norm, dim=1)                                                        # Compute the norm for each vector difference  ||W_theta * (x_i - x_j)||
                list_inc_score[index_current_node] = (self.W_phi * neigh_distances.max()).clone()                   # Add (W_phi * ||W_theta * (x_i - x_j)||) to the inclusion score list
            # print('with loop : ', list_inc_score.shape) 

        if len(previous_inclusion_score)==0:                            # return if no previous inclusion score
            if return_flatten:
                list_inc_score = list_inc_score.flatten()
            return list_inc_score
        
        if list_inc_score.shape[1]!=1:                                  # If inclusion score is not vector
            list_inc_score = torch.mean(list_inc_score, dim=1)            # Mean the matrix for each node

        # array of array to array
        if len(list_inc_score.shape)==2:                 
            if list_inc_score.shape[1]==1:
                list_inc_score = list_inc_score.flatten()

        result_np = torch.stack([previous_inclusion_score, list_inc_score])
        
        result_np = torch.mean(result_np, dim=0)
        
        return result_np