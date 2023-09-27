############################################# LES IMPORTS PyTorch PyG ######################################################
# Installer les packages requis
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# Assurez-vous que PyTorch et PyG ont la même version
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-geometric

# Import des bibliothèques
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv


######################################## Construction de notre GCN ########################################################
# Charger le jeu de données KarateClub
dataset = KarateClub()
data = dataset[0]

# Définition du modèle GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)  # Utilisation de 16 neurones pour la première couche
        self.conv2 = GCNConv(16, dataset.num_classes)  # Utilisation de num_classes neurones pour la sortie

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Fonction d'activation ReLU entre les couches
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout pour régularisation
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Log-softmax pour la classification

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
model = GCN()
criterion = torch.nn.NLLLoss()  # Negative log likelihood loss pour la classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entraînement du modèle
def train(data):
    model.train()
    optimizer.zero_grad()  # Effacer les gradients
    out = model(data.x, data.edge_index)  # Propagation avant
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Calcul de la perte
    loss.backward()  # Rétropropagation des gradients
    optimizer.step()  # Mise à jour des paramètres
    return loss.item()

# Boucle d'entraînement
for epoch in range(100):
    loss = train(data)
    print(f'Epoch: {epoch}, Loss: {loss:.4f}')

# Évaluation du modèle
#model.eval()
#pred = model(data.x, data.edge_index).max(dim=1)[1]
#...
#print(f'Test Accuracy: {acc:.4f}')
