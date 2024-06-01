import torch
import numpy as np

from igraph import Graph as igraphGraph
from loss.probabilistic_chamfer_distance import torch_d_P_Ps
from loss.probabilistic_surfaces_distance import torch_d_f_S_Ss, torch_d_r_S_Ss


def total_loss(inclusion_score, graph_nodes, extended_graph_nodes, final_scores, selected_triangles, selected_triangles_indexes, graph):
    d_P_Ps = torch_d_P_Ps(inclusion_score, graph_nodes, extended_graph_nodes)

    igraph_g_original = igraphGraph(directed=False).from_networkx(graph)
    triangles_ids_igraph_original = np.array(igraph_g_original.cliques(min=3, max=3))
    triangles_original = np.array(igraph_g_original.vs['_nx_name'])[triangles_ids_igraph_original]
    b = torch.Tensor(np.mean(triangles_original, axis=1)).to("cuda" if torch.cuda.is_available() else "cpu")

    b_hat = selected_triangles.mean(dim=1)

    p_b_hat = final_scores[selected_triangles_indexes]

    d_f_S_Ss = torch_d_f_S_Ss(p_b_hat, b_hat, b)

    d_f_S_Ss = torch_d_r_S_Ss(final_scores, p_b_hat, b, b_hat)

    loss = d_P_Ps + d_f_S_Ss + d_f_S_Ss

    return loss