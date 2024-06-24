import torch
import numpy as np

from igraph import Graph as igraphGraph
from loss.probabilistic_chamfer_distance import torch_d_P_Ps
from loss.probabilistic_surfaces_distance import torch_d_f_S_Ss, torch_d_r_S_Ss


def total_loss(inclusion_score, graph_nodes, extended_graph_nodes, selected_triangles_probabilities, selected_barycenters, original_barycenters):
    d_P_Ps = torch_d_P_Ps(inclusion_score, graph_nodes, extended_graph_nodes)

    b = original_barycenters

    b_hat = selected_barycenters.mean(dim=1)

    p_b_hat = selected_triangles_probabilities

    d_f_S_Ss = torch_d_f_S_Ss(p_b_hat, b_hat, b)

    # d_f_S_Ss = torch_d_r_S_Ss(final_scores, p_b_hat, b, b_hat)

    loss = d_P_Ps + d_f_S_Ss #+ d_f_S_Ss

    return loss