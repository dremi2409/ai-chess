from ai_chess.engine import play_random_move
import torch

print("Torch GPU disponible :", torch.cuda.is_available())

# Petit test de partie aléatoire
print("\n=== Début de partie aléatoire ===")
play_random_move()
