import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from model.mesh_dataset import MeshDataset
from model.model_point_picker import ModelPointPicker

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

number_neigh_tri = 20
torch_dataset = MeshDataset("3d_models/stl/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



gnn_model = ModelPointPicker().to(device)
gnn_model.load_state_dict(torch.load('save_models/point_picker/1/720.pt'))

points = torch_dataset[0][0].x.cpu()

for i in range(0,300,10):
    fig = plt.figure()

    s = [1 for i in range(len(points))]
    ax = fig.add_subplot(121,projection='3d')
    ax.view_init(elev = 110+i, azim=-90)
    ax.scatter(points[:,0],points[:,1],points[:,2], s=s)


    s = [1 for i in range(len(points))]
    ax = fig.add_subplot(122,projection='3d')
    ax.view_init(elev = 110+i, azim=-90)
    ax.scatter(points[:,0],points[:,1],points[:,2])
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# saved_model_filenames = [f for f in os.listdir('save_models/')]
# for save_model in tqdm(saved_model_filenames):
#         gnn_model = ModelPointPicker(number_neigh_tri).to(device)
#         gnn_model.load_state_dict(torch.load('save_models/'+save_model))
#         gnn_model.eval()
#         with torch.no_grad():