import torch
import time
import random

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model.mesh_dataset import MeshDataset
from model.model_triangle_generator import ModelTriangleGenerator
from model.loss.loss_triangle_generator import triangle_generator_loss

writer = SummaryWriter()

torch.manual_seed(42)

max_epoch = 20_000_000_000
min_number_points = 60
max_number_points = 15_000
number_neigh_tri = 20
save_model_every = 20       # batch


def train():
    torch_dataset = MeshDataset("3d_models/stl/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_triangle_generator = ModelTriangleGenerator(number_neigh_tri).to(device)
    optimizer = torch.optim.Adam(model_triangle_generator.parameters(), lr=1e-3)

    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=10),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=False
    # ) as prof:
    for epoch in range(max_epoch): 
        d_f_S_Ss_epoch, d_r_S_Ss_epoch = 0.0, 0.0
        simplification_rate = random.random() * 0.8     

        for i, (torch_graph, original_triangles) in enumerate(torch_dataset):
            target_number_point = int(max(min_number_points, min(len(torch_graph.x)*simplification_rate, max_number_points)))

            selected_triangles = model_triangle_generator(target_number_point, torch_graph)
            
            d_f_S_Ss, d_r_S_Ss = triangle_generator_loss(torch_graph.x, 
                                                        model_triangle_generator.original_barycenters,
                                                        selected_triangles, 
                                                        model_triangle_generator.selected_triangles_probabilities)
            d_f_S_Ss_epoch += d_f_S_Ss
            d_r_S_Ss_epoch += d_r_S_Ss
        
        total_loss_epoch = d_f_S_Ss_epoch + d_r_S_Ss_epoch
        total_loss_epoch.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar('Loss/total', total_loss_epoch, epoch)
        writer.add_scalar('Loss/d_f_S_Ss', d_f_S_Ss_epoch, epoch)
        writer.add_scalar('Loss/d_r_S_Ss', d_r_S_Ss_epoch, epoch)
        print('EPOCH : ', epoch, '   |   SIMPLIFICATION : ', str(int(simplification_rate*100)),'%  |   LOSS : ', total_loss_epoch.data.item())

        if epoch%save_model_every==0:
            torch.save(model_triangle_generator.state_dict(), 'save_models/triangle_generator/1/'+str(epoch)+'.pt')

train()