import chess
import random

def play_random_move():
    board = chess.Board()
    move_number = 1
    while not board.is_game_over():
        move_number += 1
        move = random.choice(list(board.legal_moves))
        board.push(move)
        print(f"Move {move_number//2}:")
        print(board, "\n")

    print("Résultat :", board.result())
