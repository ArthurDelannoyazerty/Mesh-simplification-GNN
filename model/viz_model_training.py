import torch
import os
import numpy as np
import vtkplotlib as vpl

from tqdm import tqdm
from stl import mesh
from mesh_dataset import MeshDataset
from gnn_simplification_model import GNNSimplificationMesh

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def viz_model_train():

    number_neigh_tri = 20
    torch_dataset = MeshDataset("3d_models/stl/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    saved_model_filenames = [f for f in os.listdir('save_models/')]
    for save_model in tqdm(saved_model_filenames):
        gnn_model = GNNSimplificationMesh(number_neigh_tri).to(device)
        gnn_model.load_state_dict(torch.load('save_models/'+save_model))
        gnn_model.eval()
        with torch.no_grad():

            index_img = int(int(save_model[2:-3])/10)

            torch_graph, triangles = torch_dataset[0]
            triangles = gnn_model(200, torch_graph)
            # triangles = triangles.detach().cpu().numpy()
            vertices = triangles.reshape(-1, 3).cpu().numpy()

            # Create a list of triangle indices
            num_triangles = triangles.shape[0]
            faces = np.arange(num_triangles * 3).reshape(-1, 3)


            # Center and scale the mesh
            centroid = np.mean(vertices, axis=0)
            vertices -= centroid  # Center the mesh
            max_distance = np.max(np.linalg.norm(vertices, axis=1))
            vertices /= max_distance  # Scale the mesh


            generated_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    generated_mesh.vectors[i][j] = vertices[f[j],:]
            
            
            

            # fig = vpl.figure()
            # angle = index_img%360
            # fig.camera.SetPosition(np.cos(np.radians(angle)) * 5, np.sin(np.radians(angle)) * 5, 5)
            # fig.camera.SetFocalPoint(1, 0, 0)
            # fig.render()

            vpl.plot(generated_mesh.vectors, join_ends=True, color="dark red")
            vpl.mesh_plot(generated_mesh)
            vpl.text(str(index_img), color=(0,0,0))

            vpl.show()
            # vpl.save_fig('saved_img/model_viz/1/'+str(index_img)+'.png', off_screen=True, pixels=(1080,1080))
            vpl.close()
            



viz_model_train()