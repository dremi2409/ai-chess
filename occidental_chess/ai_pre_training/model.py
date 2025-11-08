import torch
import torch.nn as nn

class ChessEvaluationNet(nn.Module):
    """
    Réseau de neurones pour évaluer une position d'échecs
    Entrée: tenseur de 71 valeurs (64: plateau aplati;  7: paramètres supplémentaires)
    Sortie: une valeur entre -1 et 1 (évaluation de la position)
    """
    def __init__(self):
        super(ChessEvaluationNet, self).__init__()
        
        # Couche d'entrée vers première couche cachée (64 -> 200)
        self.fc1 = nn.Linear(71, 200)
        
        # Première couche cachée vers deuxième couche cachée (200 -> 200)
        self.fc2 = nn.Linear(200, 200)
        
        # Deuxième couche cachée vers sortie (200 -> 1)
        self.fc3 = nn.Linear(200, 1)
        
        # Fonction d'activation ReLU
        self.relu = nn.ReLU()
        
        # Fonction d'activation Tanh pour la sortie (entre -1 et 1)
        self.tanh = nn.Tanh()
        
        # Afficher le nombre de paramètres
        self._print_num_parameters()
    
    def _print_num_parameters(self):
        """Affiche le nombre total de paramètres du réseau"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*50}")
        print(f"Nombre total de paramètres : {total_params:,}")
        print(f"Paramètres entraînables : {trainable_params:,}")
        print(f"{'='*50}\n")
    
    def forward(self, x):
        # Première couche cachée avec ReLU
        x = self.relu(self.fc1(x))
        
        # Deuxième couche cachée avec ReLU
        x = self.relu(self.fc2(x))
        
        # Couche de sortie avec Tanh
        x = self.tanh(self.fc3(x))
        
        return x