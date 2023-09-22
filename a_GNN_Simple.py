import torch
import torch.nn as nn
import torch.optim as optim

# Définition des données d'entraînement 
num_nodes = 5  # Nombre de nœuds dans le graphe
feat_dim = 3   # Dimension des caractéristiques des nœuds
num_classes = 2  # Nombre de classes pour la classification
num_epochs = 100  # Nombre d'époques d'entraînement

# Caractéristiques des nœuds et étiquettes
node_features = torch.randn(num_nodes, feat_dim)
node_labels = torch.LongTensor([0, 1, 0, 1, 1])

# Matrice d'adjacence 
adjacency_matrix = torch.tensor([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
], dtype=torch.float32)

########################   Définition du modèle GNN simple ##############################################
class GNNSimple(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GNNSimple, self).__init__()
        self.gc1 = nn.Linear(in_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, adjacency_matrix, node_features):
        # Propagation avant avec une couche GCN
        x = torch.matmul(adjacency_matrix, node_features)
        x = torch.relu(x)
        x = self.gc1(x)
        x = torch.relu(x)

        # Nouvelle propagation en utilisant la matrice d'adjacence mise à jour
        x = torch.matmul(adjacency_matrix, x)
        x = self.gc2(x)
        return x

########################### Création du modèle ##############################################
model = GNNSimple(in_dim=feat_dim, hidden_dim=16, out_dim=num_classes)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

########################### Training our model ##############################################
for epoch in range(num_epochs):
    model.train()  # Mettre le modèle en mode d'entraînement

    # Propagation avant
    output = model(adjacency_matrix, node_features)

    # Calcul de la perte
    loss = criterion(output, node_labels)

    # Rétropropagation et mise à jour des poids
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Affichage de la perte à chaque époque (facultatif)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

