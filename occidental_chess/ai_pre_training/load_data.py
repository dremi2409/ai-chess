import torch
from torch.utils.data import Dataset
import numpy as np

class ChessDataset(Dataset):
    """Dataset personnalisé pour les positions d'échecs"""
    def __init__(self, data_path):
        # Charger les données
        self.data = np.load(data_path)
        print(f"Dataset chargé : {self.data.shape[0]} positions")
        
        # Séparer les features (71 premières colonnes) et les labels (dernière colonne)
        self.features = torch.FloatTensor(self.data[:, :71])
        self.labels = torch.FloatTensor(self.data[:, 71:72])  # Garder la dimension
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class ChessDataset2(Dataset):
    """Dataset personnalisé pour les positions d'échecs"""
    def __init__(self, positions):
        # Charger les positions numpy 
        self.data = np.array(positions)
        #print(f"Dataset chargé : {self.data.shape[0]} positions")
        self.features = torch.FloatTensor(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx]