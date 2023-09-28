#install required packages
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
#ensure that the PyTorch and the PyG are the same version

    ################### PIP INSTALL SUR LE TERMINAL ####################
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

#################### IMPORTS#################################"""
# Import des bibliothèques nécessaires


# Helper function for visualization.
%matplotlib inline
import networkx as nx
import matplotlib.pyplot as plt

#to implement our graph neural network
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#######################Initialisation de notre dataset #################################
from torch_geometric.datasets import KarateClub
dataset = KarateClub()
print('Dataset properties')
print('==============================================================')
print(f'Dataset: {dataset}') #This prints the name of the dataset
print(f'Number of graphs in the dataset: {len(dataset)}')
print(f'Number of features: {dataset.num_features}') #Number of features each node in the dataset has
print(f'Number of classes: {dataset.num_classes}') #Number of classes that a node can be classified into


#Since we have one graph in the dataset, we will select the graph and explore it's properties

data = dataset[0]
print('Graph properties')
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}') #Number of nodes in the graph
print(f'Number of edges: {data.num_edges}') #Number of edges in the graph
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # Average number of nodes in the graph
print(f'Contains isolated nodes: {data.has_isolated_nodes()}') #Does the graph contains nodes that are not connected
print(f'Contains self-loops: {data.has_self_loops()}') #Does the graph contains nodes that are linked to themselves
print(f'Is undirected: {data.is_undirected()}') #Is the graph an undirected graph


######################################### VISUALISATION DE NOTRE GRAPHE ################################
# Importation des bibliothèques nécessaires
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx  # Assurez-vous que NetworkX est installé

# Définition d'une fonction pour visualiser un graphe
def visualize_graph(G, color):
    # Crée une figure pour la visualisation du graphe
    plt.figure(figsize=(7, 7))
    
    # Masque les marques sur les axes x et y
    plt.xticks([])
    plt.yticks([])
    
    # Utilise l'algorithme de disposition spring_layout pour organiser les nœuds du graphe
    # (seed=6 pour la reproductibilité de la disposition)
    pos = nx.spring_layout(G, seed=6)
    
    # Dessine le graphe en utilisant NetworkX
    nx.draw_networkx(G, pos=pos, with_labels=False, node_color=color, cmap="Set2")
    
    # Affiche la visualisation du graphe
    plt.show()

# Convertit le graphe PyTorch Geometric (data) en un graphe NetworkX
# avec l'option to_undirected=True pour obtenir un graphe non orienté
G = to_networkx(data, to_undirected=True)

# Appelle la fonction pour visualiser le graphe avec des couleurs basées sur les étiquettes (data.y)
visualize_graph(G, color=data.y)

############################### Implementing a graph neural network ##################################""
# Définition de la classe GCN (Graph Convolutional Network)
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()  # Appel du constructeur de la classe parente (torch.nn.Module)
        torch.manual_seed(12345)  # Fixer la graine pour la reproductibilité des résultats

        # Définition des couches de convolution graphique (GCN) et de la couche de classification linéaire
        self.conv1 = GCNConv(dataset.num_features, 4)  # Première couche GCN, prend le nombre de caractéristiques du jeu de données en entrée et produit 4 caractéristiques en sortie
        self.conv2 = GCNConv(4, 4)  # Deuxième couche GCN, prend 4 caractéristiques en entrée et produit à nouveau 4 caractéristiques en sortie
        self.conv3 = GCNConv(4, 2)  # Troisième couche GCN, prend 4 caractéristiques en entrée et produit 2 caractéristiques en sortie
        
        # Utilisation de la fonction d'activation tanh pour introduire de la non-linéarité entre les couches
        self.tanh = torch.nn.Tanh()

        self.classifier = Linear(2, dataset.num_classes)  # Couche de classification linéaire, prend 2 caractéristiques en entrée et produit un nombre de caractéristiques en sortie égal au nombre de classes dans le jeu de données

    # Définition de la méthode forward pour spécifier comment les données sont propagées à travers le modèle
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)  # Propagation à travers la première couche GCN
        h = self.tanh(h)  # Application de la fonction d'activation tanh pour introduire de la non-linéarité
        h = self.conv2(h, edge_index)  # Propagation à travers la deuxième couche GCN
        h = self.tanh(h)  # Application de la fonction d'activation tanh pour introduire de la non-linéarité
        h = self.conv3(h, edge_index)  # Propagation à travers la troisième couche GCN
        h = self.tanh(h)  # Application de la fonction d'activation tanh pour obtenir l'espace d'incorporation GNN final
        
        # Application de la couche de classification linéaire pour la prédiction
        out = self.classifier(h)

        return out, h  # Retourne la sortie de classification et l'espace d'incorporation GNN final

# Création d'une instance du modèle GCN
model = GCN()

# Affichage du modèle
print(model)

 ################# Entrainement de notre model #######################
model = GCN()
criterion = torch.nn.CrossEntropyLoss()  #Initialize the CrossEntropyLoss function.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Initialize the Adam optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    print(f'Epoch: {epoch}, Loss: {loss}')











