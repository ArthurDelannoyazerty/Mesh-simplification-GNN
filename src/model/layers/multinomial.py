import torch
import torch.nn as nn

class MultinomialLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, target_number_points, nodes):
        normalized_inclusion_score = f / torch.sum(f)                           # normalize for multinomial sampling

        mult_sampling = torch.distributions.multinomial.Multinomial(total_count=10*normalized_inclusion_score.shape[0], probs=normalized_inclusion_score).sample()      # small:more randomness    |   big:less randomness
        mult_indices = mult_sampling.topk(k=target_number_points).indices
        selected_nodes = nodes[mult_indices]

        return selected_nodes