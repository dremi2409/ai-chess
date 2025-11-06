import matplotlib.pyplot as plt
import numpy as np
import time
import os

def plot_losses():
    """Plot les courbes de loss en temps réel"""
    
    LOSS_FILE = "results/loss.dat"
    TEST_FILE = "results/test_error.dat"
    SLEEP_TIME = 5  # secondes entre chaque actualisation
    
    # Créer la figure
    plt.ion()  # Mode interactif
    fig, ax = plt.subplots(figsize=(12, 6))
    
    print("Démarrage du monitoring des losses...")
    print(f"Actualisation toutes les {SLEEP_TIME} secondes")
    print("Appuyez sur Ctrl+C pour arrêter\n")
    
    try:
        while True:
            # Vérifier si les fichiers existent
            if not os.path.exists(LOSS_FILE):
                print(f"En attente du fichier {LOSS_FILE}...")
                time.sleep(SLEEP_TIME)
                continue
            
            # Charger les données
            try:
                train_data = np.loadtxt(LOSS_FILE, skiprows=1)
                
                # Vérifier si le fichier de test existe
                test_exists = os.path.exists(TEST_FILE)
                if test_exists:
                    test_data = np.loadtxt(TEST_FILE, skiprows=1)
                
                # Effacer le graphique précédent
                ax.clear()
                
                # Plot train loss
                if train_data.ndim == 1:
                    # Un seul point
                    ax.plot(train_data[0], train_data[1], 'b-', label='Train Loss', linewidth=2)
                else:
                    ax.plot(train_data[:, 0], train_data[:, 1], 'b-', label='Train Loss', linewidth=2)
                
                # Plot test loss si disponible
                if test_exists:
                    if test_data.ndim == 1:
                        ax.plot(test_data[0], test_data[1], 'ro-', label='Test Loss', 
                               linewidth=2, markersize=6)
                    else:
                        ax.plot(test_data[:, 0], test_data[:, 1], 'ro-', label='Test Loss', 
                               linewidth=2, markersize=6)
                
                # Configuration du graphique
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Loss (L1)', fontsize=12)
                ax.set_title('Évolution des losses pendant l\'entraînement', 
                           fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # Afficher les dernières valeurs
                if train_data.ndim == 1:
                    last_epoch = int(train_data[0])
                    last_train = train_data[1]
                else:
                    last_epoch = int(train_data[-1, 0])
                    last_train = train_data[-1, 1]
                
                info_text = f"Epoch {last_epoch} - Train: {last_train:.6f}"
                
                if test_exists:
                    if test_data.ndim == 1:
                        last_test = test_data[1]
                    else:
                        last_test = test_data[-1, 1]
                    info_text += f" - Test: {last_test:.6f}"
                
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)
                
            except Exception as e:
                print(f"Erreur lors du chargement des données : {e}")
            
            # Attendre avant la prochaine actualisation
            time.sleep(SLEEP_TIME)
            
    except KeyboardInterrupt:
        print("\n\nArrêt du monitoring.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    plot_losses()