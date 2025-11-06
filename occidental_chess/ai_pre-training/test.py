import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model import ChessEvaluationNet
from load_data import ChessDataset

def test_model(model, test_loader, criterion, device):
    """Évalue le modèle sur l'ensemble de test"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            # Déplacer les données sur le device
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Calculer la loss
            loss = criterion(outputs, labels)
            
            # Accumuler la loss
            total_loss += loss.item()
            num_batches += 1
            
            # Sauvegarder les prédictions et labels pour analyse
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    # Concaténer toutes les prédictions et labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return avg_loss, all_predictions, all_labels


def calculate_metrics(predictions, labels):
    """Calcule des métriques supplémentaires"""
    # Erreur absolue moyenne (MAE) - identique à L1 loss
    mae = np.mean(np.abs(predictions - labels))
    
    # Erreur quadratique moyenne (MSE)
    mse = np.mean((predictions - labels) ** 2)
    
    # Racine de l'erreur quadratique moyenne (RMSE)
    rmse = np.sqrt(mse)
    
    # Erreur maximale
    max_error = np.max(np.abs(predictions - labels))
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'max_error': max_error
    }


def main():
    # Paramètres
    BATCH_SIZE = 10000
    TEST_DATA_PATH = "../dataset/test_positions.npy"
    MODEL_PATH = "checkpoints/model_final.pth"  # Vous pouvez changer pour un autre checkpoint
    
    # Détecter si GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")
    
    # Charger le dataset de test
    print(f"\nChargement du dataset de test...")
    test_dataset = ChessDataset(TEST_DATA_PATH)
    
    # Créer le DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Pas de mélange pour le test
        num_workers=0
    )
    
    # Initialiser le modèle
    model = ChessEvaluationNet().to(device)
    
    # Charger les poids entraînés
    print(f"Chargement du modèle depuis : {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modèle chargé (entraîné jusqu'à l'epoch {checkpoint['epoch']})")
    
    # Définir la fonction de loss (L1 Loss)
    criterion = nn.L1Loss()
    
    print(f"\n{'='*70}")
    print(f"Début de l'évaluation sur l'ensemble de test")
    print(f"{'='*70}\n")
    
    # Évaluer le modèle
    avg_loss, predictions, labels = test_model(model, test_loader, criterion, device)
    
    # Calculer des métriques supplémentaires
    metrics = calculate_metrics(predictions, labels)
    
    # Afficher les résultats
    print(f"{'='*70}")
    print(f"RÉSULTATS DE L'ÉVALUATION")
    print(f"{'='*70}")
    print(f"Nombre de positions testées : {len(test_dataset):,}")
    print(f"\nLoss L1 (MAE) : {avg_loss:.6f}")
    print(f"MSE           : {metrics['mse']:.6f}")
    print(f"RMSE          : {metrics['rmse']:.6f}")
    print(f"Erreur max    : {metrics['max_error']:.6f}")
    print(f"{'='*70}\n")
    
    # Afficher quelques exemples de prédictions
    print("Exemples de prédictions (10 premières positions) :")
    print(f"{'Prédiction':<15} {'Valeur réelle':<15} {'Erreur':<15}")
    print("-" * 45)
    for i in range(min(34, len(predictions))):
        pred = predictions[i][0]
        label = labels[i][0]
        error = abs(pred - label)
        print(f"{pred:>14.6f} {label:>14.6f} {error:>14.6f}")


if __name__ == "__main__":
    main()