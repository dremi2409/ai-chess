import chess
import torch
import random
import numpy as np

def board_to_numpy(board):
    """
    Convertit un plateau chess.Board en array numpy 8x8
    avec des valeurs numériques pour chaque pièce
    """
    # Mapping des pièces vers des valeurs numériques
    piece_to_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6
    }
    
    # Initialiser un array 8x8 avec des zéros
    board_array = np.zeros((8, 8), dtype=int)
    
    # Parcourir toutes les cases du plateau
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Calculer la position dans l'array (rank, file)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            # Obtenir la valeur de la pièce
            value = piece_to_value[piece.piece_type]
            
            # Négatif pour les noirs, positif pour les blancs
            if piece.color == chess.BLACK:
                value = -value
            
            board_array[7 - rank, file] = value  # 7-rank pour inverser (rank 0 en bas)
    
    return board_array

def board_to_tensor(board):
    """
    Convertit un plateau chess.Board en tenseur PyTorch normalisé
    """
    board_array = board_to_numpy(board)
    board_flat = board_array.flatten()
    board_normalized = board_flat / 6.0 # Normalise entre -1 et 1
    board_tensor = torch.tensor(board_normalized, dtype=torch.float32)
    return board_tensor

def play_random_move():
    board = chess.Board()
    move_number = 1
    while not board.is_game_over() and move_number <= 10:
        move_number += 1
        move = random.choice(list(board.legal_moves))
        print("Liste de coups légaux :", list(board.legal_moves))
        print("move choisi :", move)
        board.push(move)
        print(f"Move {move_number//2}:")
        print(board, "\n")
        board_array = board_to_numpy(board)
        print("Représentation numpy du plateau :")
        print(board_array)

    print("Résultat :", board.result())
