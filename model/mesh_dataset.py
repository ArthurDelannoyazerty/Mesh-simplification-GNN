import os

from torch.utils.data import Dataset, DataLoader
from transformation import Transformation

class MeshDataset(Dataset):
    def __init__(self, mesh_dir):
        self.mesh_dir = mesh_dir
        self.filepaths = list()
        for filename in os.listdir(mesh_dir):
            f = os.path.join(mesh_dir, filename)
            if os.path.isfile(f):
                self.filepaths.append(f)
        self.len = len(self.filepaths)
        self.transormation = Transformation()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mesh_path = self.filepaths[idx]
        mesh_data = self.transformation.stl_to_mesh(mesh_path)
        graph = self.transformation.mesh_to_graph(mesh_data)
        return mesh_data, graph