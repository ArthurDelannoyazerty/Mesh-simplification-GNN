import torch
import torch.nn as nn

class MultinomialLayer(nn.Module):
    def __init__(self, target_number_points,nodes):
        super().__init__()
        self.target_number_points = target_number_points
        self.nodes = nodes

    def forward(self, f):
        normalized_inclusion_score = f / torch.sum(f)                           # normalize for multinomial sampling

        mult_sampling = torch.distributions.multinomial.Multinomial(total_count=10*normalized_inclusion_score.shape[0], probs=normalized_inclusion_score).sample()      # small:more randomness    |   big:less randomness
        mult_indices = mult_sampling.topk(k=self.target_number_points).indices
        selected_nodes = self.nodes[mult_indices]

        return selected_nodes