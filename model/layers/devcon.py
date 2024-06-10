import torch
import torch.nn as nn

class DevConv(nn.Module):
    def __init__(self, output_dimension):
        super().__init__()
        self.size = output_dimension

        self.W_phi = nn.Parameter(torch.Tensor(output_dimension))
        self.W_theta = nn.Parameter(torch.Tensor(size=(3,output_dimension)))
        nn.init.normal_(self.W_phi)
        nn.init.normal_(self.W_theta)
    
    
    def forward(self, previous_inclusion_score, nodes, adjacency_matrix, return_flatten=True):
        
        indices = adjacency_matrix.coalesce().indices()
        distances = nodes[indices[1]] - nodes[indices[0]]
        w_theta_operation = (distances @ self.W_theta).squeeze()
        non_padded_grouped_distance = w_theta_operation.split(tuple(torch.bincount(indices[0])))
        padded_grouped_distance = torch.nn.utils.rnn.pad_sequence(non_padded_grouped_distance, batch_first=True, padding_value=-torch.inf)
        maxi = padded_grouped_distance.max(dim=1).values
        list_inc_score = self.W_phi * maxi
            
        if len(previous_inclusion_score)==0:                            # return if no previous inclusion score
            if return_flatten:
                list_inc_score = list_inc_score.flatten()
            return list_inc_score
        
        if len(list_inc_score.shape)!=1:
            if list_inc_score.shape[1]!=1:
                list_inc_score = torch.mean(list_inc_score, dim=1)          # Mean the inclusion score (hidden dimension)

        return previous_inclusion_score + list_inc_score