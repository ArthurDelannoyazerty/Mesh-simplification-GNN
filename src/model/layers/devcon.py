import torch
import torch.nn as nn
import time
import torchviz

class DevConv(nn.Module):
    def __init__(self, output_dimension):
        super().__init__()
        self.size = output_dimension

        self.W_phi = nn.Parameter(torch.Tensor(output_dimension))
        self.W_theta = nn.Parameter(torch.Tensor(size=(3,output_dimension)))
        nn.init.normal_(self.W_phi)
        nn.init.normal_(self.W_theta)
    
    
    def forward(self, previous_inclusion_score, nodes, adjacency_matrix, return_flatten=True):
        
        # SLOW
        # indices = adjacency_matrix.coalesce().indices()
        # distances = nodes[indices[1]] - nodes[indices[0]]
        # w_theta_operation = (distances @ self.W_theta).squeeze()
        # non_padded_grouped_distance = w_theta_operation.split(tuple(torch.bincount(indices[0])))
        # padded_grouped_distance = torch.nn.utils.rnn.pad_sequence(non_padded_grouped_distance, batch_first=True, padding_value=-torch.inf)
        # max_distances = padded_grouped_distance.max(dim=1).values
        
        
        # display(torchviz.make_dot(non_padded_grouped_distance))

        # FAST
        dist = nodes.unsqueeze(0) - nodes.unsqueeze(1)
        wo = (dist @ self.W_theta).squeeze()
        masked_distances = wo * adjacency_matrix.float()
        # masked_distances = masked_distances.to_dense()
        # max_distances2, _ = torch.max(masked_distances, dim=1)
        # display(torchviz.make_dot(masked_distances))
        max_distances = torch.amax(masked_distances, dim=1)
        torch.Tensor.sparse_resize_()
        
        
        list_inc_score = self.W_phi * max_distances
        
        if len(previous_inclusion_score)==0:                            # return if no previous inclusion score
            if return_flatten:
                list_inc_score = list_inc_score.flatten()
            return list_inc_score
        
        if len(list_inc_score.shape)!=1:
            if list_inc_score.shape[1]!=1:
                list_inc_score = torch.mean(list_inc_score, dim=1)          # Mean the inclusion score (hidden dimension)

        return previous_inclusion_score + list_inc_score