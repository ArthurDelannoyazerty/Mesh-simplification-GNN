import torch
import time

from mesh_dataset import MeshDataset
from torch.utils.data import DataLoader
from gnn_simplification_model import GNNSimplificationMesh
from loss.loss import total_loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

torch.manual_seed(42)


def train():
    number_neigh_tri = 20

    # if len(graph._node)<20:
    #     raise Exception("Input mesh does not have enough vertices. (More than 20 is needed)")

    torch_dataset = MeshDataset("3d_models/stl/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model = GNNSimplificationMesh(number_neigh_tri).to(device)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-5)

    
    for epoch in range(5000000): 
        d_P_Ps_epoch, d_f_S_Ss_epoch = 0.0, 0.0
        
        for i, (torch_graph, triangles) in enumerate(torch_dataset):
            selected_triangles = gnn_model(200, torch_graph)
            
            d_P_Ps, d_f_S_Ss = total_loss(gnn_model.score_original_points, 
                              torch_graph.x, 
                              gnn_model.generated_graph_nodes, 
                              gnn_model.selected_triangles_probabilities, 
                              selected_triangles, 
                              gnn_model.original_barycenters)
            d_P_Ps_epoch   += d_P_Ps
            d_f_S_Ss_epoch += d_f_S_Ss
        
        d_P_Ps_epoch.backward() 
        d_f_S_Ss_epoch.backward() 
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss = d_P_Ps_epoch + d_f_S_Ss_epoch
        writer.add_scalar('Loss/total', epoch_loss, epoch)
        writer.add_scalar('Loss/Point-Sampler', d_P_Ps_epoch, epoch)
        writer.add_scalar('Loss/Triangle-Generation', d_f_S_Ss_epoch, epoch)
        print('EPOCH : ', epoch, '   |   LOSS : ', epoch_loss.data.detach().cpu().numpy())

        if epoch%10==0:
            torch.save(gnn_model.state_dict(), 'save_models/'+str(epoch)+'.pt')

train()