import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from model import ChessEvaluationNet
from load_data import ChessDataset


def evaluate_test_set(model, test_loader, criterion, device):
    """Évalue le modèle sur l'ensemble de test"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(features)
    
    return total_loss / len(test_loader.dataset)


def train_model(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, labels in train_loader:
        # Déplacer les données sur le device (CPU ou GPU)
        features = features.to(device)
        labels = labels.to(device)
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        
        # Calculer la loss
        loss = criterion(outputs, labels)
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        
        # Accumuler la loss
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    # Paramètres d'entraînement
    BATCH_SIZE = 600000
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.001
    SAVE_INTERVAL = 20
    TEST_INTERVAL = 5
    TRAIN_DATA_PATH = "../dataset/train_positions.npy"
    TEST_DATA_PATH = "../dataset/test_positions.npy"
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    
    # Créer les dossiers
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Détecter si GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")
    
    # Charger les datasets
    train_dataset = ChessDataset(TRAIN_DATA_PATH)
    test_dataset = ChessDataset(TEST_DATA_PATH)
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=6
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # Tout le dataset de test en un seul batch
        shuffle=False,
        num_workers=0
    )
    
    # Initialiser le modèle
    model = ChessEvaluationNet().to(device)
    
    # Définir la fonction de loss (L1 Loss = norme 1)
    criterion = nn.L1Loss()
    
    # Définir l'optimiseur
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n{'='*70}")
    print(f"Début de l'entraînement")
    print(f"{'='*70}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Nombre d'epochs : {NUM_EPOCHS}")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Sauvegarde tous les {SAVE_INTERVAL} epochs")
    print(f"Test tous les {TEST_INTERVAL} epochs")
    print(f"{'='*70}\n")
    
    # Listes pour stocker les losses
    train_losses = []
    test_losses = []
    test_epochs = []
    
    # Variables pour l'early stopping
    last_three_test_losses = []
    
    # Boucle d'entraînement
    for epoch in range(1, NUM_EPOCHS + 1):
        # Entraîner pour une epoch
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Afficher la train loss
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Train Loss : {train_loss:.6f}", end="")
        
        # Tester tous les TEST_INTERVAL epochs
        if epoch % TEST_INTERVAL == 0:
            # Évaluer sur le dataset de test
            test_loss = evaluate_test_set(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_epochs.append(epoch)
            
            print(f" - Test Loss : {test_loss:.6f}")
            
            # Sauvegarder les losses dans les fichiers
            loss_file = os.path.join(RESULTS_DIR, "loss.dat")
            test_file = os.path.join(RESULTS_DIR, "test_error.dat")
            
            # Sauvegarder toutes les train losses avec leur epoch
            np.savetxt(loss_file, 
                      np.column_stack((np.arange(1, len(train_losses) + 1), train_losses)),
                      fmt='%d %.6f',
                      header='epoch train_loss',
                      comments='')
            
            # Sauvegarder toutes les test losses avec leur epoch
            np.savetxt(test_file,
                      np.column_stack((test_epochs, test_losses)),
                      fmt='%d %.6f',
                      header='epoch test_loss',
                      comments='')
            
            # Vérifier l'early stopping (3 tests consécutifs avec erreur en hausse)
            last_three_test_losses.append(test_loss)
            if len(last_three_test_losses) > 3:
                last_three_test_losses.pop(0)
            
            if len(last_three_test_losses) == 3:
                if (last_three_test_losses[1] > last_three_test_losses[0] and 
                    last_three_test_losses[2] > last_three_test_losses[1]):
                    print(f"\n{'='*70}")
                    print(f"EARLY STOPPING : L'erreur de test augmente depuis 3 tests consécutifs")
                    print(f"Arrêt de l'entraînement à l'epoch {epoch}")
                    print(f"{'='*70}\n")
                    break
        else:
            print()  # Nouvelle ligne
        
        # Sauvegarder les poids tous les SAVE_INTERVAL epochs
        if epoch % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  → Checkpoint sauvegardé : {checkpoint_path}")
    
    # Sauvegarder le modèle final
    final_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
    }, final_path)
    
    print(f"\n{'='*70}")
    print(f"Entraînement terminé !")
    print(f"Modèle final sauvegardé : {final_path}")
    print(f"Fichiers de résultats : results/loss.dat et results/test_error.dat")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()