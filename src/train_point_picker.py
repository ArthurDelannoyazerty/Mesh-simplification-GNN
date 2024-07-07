import torch
import time
import random

from model.mesh_dataset import MeshDataset
from torch.utils.data import DataLoader
from model.model_point_picker import ModelPointPicker
from model.loss.loss_point_picker import torch_d_P_Ps
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
torch.manual_seed(42)

max_epoch = 20_000_000_000
min_number_points = 1
max_number_points = 25_000
save_model_every = 20       # batch


def train():
    torch_dataset = MeshDataset("3d_models/stl/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_point_picker = ModelPointPicker().to(device)
    optimizer = torch.optim.Adam(model_point_picker.parameters(), lr=1e-3)
    
    for epoch in range(max_epoch): 
        d_P_Ps_epoch = 0.0
        simplification_rate = random.random()
        
        for i, (torch_graph, _) in enumerate(torch_dataset):
            target_number_point = int(max(min_number_points, min(len(torch_graph.x)*simplification_rate, max_number_points)))
        
            score_original_points, generated_graph_nodes = model_point_picker(target_number_point, torch_graph)
            
            d_P_Ps = torch_d_P_Ps(score_original_points, torch_graph.x, generated_graph_nodes, simplification_rate)
            d_P_Ps_epoch += d_P_Ps
        
        d_P_Ps_epoch.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar('Loss/d_P_Ps', d_P_Ps_epoch, epoch)
        print('EPOCH : ', epoch, '   |   SIMPLIFICATION : ', str(int(simplification_rate*100)),'%   |   LOSS : ', d_P_Ps.data.item())

        if epoch%save_model_every==0:
            torch.save(model_point_picker.state_dict(), 'save_models/point_picker/1/'+str(epoch)+'.pt')

train()