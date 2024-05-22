import torch
import torch.nn as nn

class DevConv(nn.Module):
    def __init__(self, nodes, adjacency_matrix, output_dimension):
        super().__init__()
        self.size = output_dimension
        self.nodes = nodes
        self.adjacency_matrix = adjacency_matrix
        self.W_phi = nn.Parameter(torch.Tensor(output_dimension))
        self.W_theta = nn.Parameter(torch.Tensor(size=(3,1)))

        nn.init.normal_(self.W_phi)
        nn.init.normal_(self.W_theta)
    
    def forward(self, previous_inclusion_score, return_flatten=True):
        list_inc_score = torch.zeros((self.nodes.shape[0], self.size))                                          #list of "output_dimension" for each "list_node" element
        for index_current_node, list_neighbors in enumerate(self.adjacency_matrix):                             # for each node and its adjacency nodes
            neighbors = self.nodes[list_neighbors.nonzero()]                                                    # get neighbors nodes
            diff = self.nodes[index_current_node] - neighbors                                                   # Compute the differences between current_node and all neighbor nodes   (x_i - x_j)
            to_norm = self.W_theta.T.unsqueeze(1).repeat(1, diff.shape[0], 1)[0] * diff.squeeze(1)              # Compute W_theta * (x_i - x_j)
            neigh_distances = torch.norm(to_norm, dim=1)                                                        # Compute the norm for each vector difference  ||W_theta * (x_i - x_j)||
            list_inc_score[index_current_node] = (self.W_phi * neigh_distances.max()).clone()                   # Add (W_phi * ||W_theta * (x_i - x_j)||) to the inclusion score list

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

        result_np = torch.stack([previous_inclusion_score, torch.tensor(list_inc_score)])
        
        result_np = torch.mean(result_np, dim=0)
        
        return result_np