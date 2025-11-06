import chess
import chess.pgn
import io
import sys
import numpy as np
from tqdm import tqdm
import os
import glob
from multiprocessing import Pool, cpu_count

def board_to_numpy(board, position_count, winner):
    """
    Convertit un plateau chess.Board en array numpy [64 + 8]
    contenant :
    - [0:64] : positions des pièces normalisées entre -1 et 1
    - [64] : à qui de jouer (+1 blanc, -1 noir)
    - [65] : compteur règle des 50 coups (normalisé entre -1 et 1)
    - [66] : droit petit roque blanc (-1 ou 1)
    - [67] : droit grand roque blanc (-1 ou 1)
    - [68] : droit petit roque noir (-1 ou 1)
    - [69] : droit grand roque noir (-1 ou 1)
    - [70] : nombre de répétitions (normalisé entre -1 et 1)
    - [71] : gagnant (+1 blanc, -1 noir, 0 nulle)
    """
    piece_to_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6
    }
    
    # Positions des pièces
    board_array = np.zeros((8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_to_value[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            board_array[7 - rank, file] = value
    
    # Aplatir les positions et normaliser
    positions = board_array.flatten() / 6.0
    
    # Informations additionnelles
    turn = 1.0 if board.turn == chess.WHITE else -1.0
    
    # Normaliser halfmove_clock entre -1 et 1 (max théorique = 100)
    halfmove_clock = (board.halfmove_clock / 50.0) - 1.0
    
    # Droits de roque : -1 si non, 1 si oui
    kingside_white = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else -1.0
    queenside_white = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else -1.0
    kingside_black = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else -1.0
    queenside_black = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else -1.0
    
    # Normaliser le nombre de répétitions entre -1 et 1 (0->-1, 1->-0.33, 2->0.33, 3->1)
    repetitions_normalized = (position_count / 1.5) - 1.0
    
    # Concaténer toutes les informations
    full_state = np.concatenate([
        positions,
        [turn, halfmove_clock, kingside_white, queenside_white, kingside_black, queenside_black, repetitions_normalized, winner]
    ])
    return full_state

def is_game_valid(game):
    """Vérifie que tous les coups du PGN sont légaux."""
    try:
        board = game.board()
        for move in game.mainline_moves():
            if not board.is_legal(move):
                return False
            board.push(move)
        return True
    except Exception:
        return False

def get_winner(game):
    """
    Détermine le gagnant de la partie.
    Retourne +1 pour blanc, -1 pour noir, 0 pour nulle
    """
    result = game.headers.get("Result", "*")
    if result == "1-0":
        return 1.0  # Blanc gagne
    elif result == "0-1":
        return -1.0  # Noir gagne
    elif result == "1/2-1/2" or result == "*":
        return 0.0  # Nulle
    else:
        return 0.0  # Résultat inconnu, traité comme nulle

def extract_positions_from_game(game):
    """Extrait toutes les positions successives d'une partie valide sous forme de vecteurs [72]."""
    positions = []
    board = game.board()
    position_history = {}  # Dictionnaire pour compter les répétitions
    
    # Déterminer le gagnant une fois pour toute la partie
    winner = get_winner(game)
    
    for move in game.mainline_moves():
        board.push(move)
        
        # Obtenir une représentation unique de la position (FEN sans les compteurs)
        fen_key = ' '.join(board.fen().split()[:4])  # Position, tour, roques, en passant
        
        # Compter les répétitions (0, 1, 2, 3+)
        if fen_key in position_history:
            position_history[fen_key] += 1
        else:
            position_history[fen_key] = 0
        
        # Limiter à 3 maximum
        repetition_count = min(position_history[fen_key], 3)
        
        positions.append(board_to_numpy(board, repetition_count, winner))

    return positions

def extract_positions_from_pgn_file(pgn_path):
    """
    Lit un fichier PGN et retourne toutes les positions avec informations complètes.
    Forme de sortie : [nb_positions, 72]
    """
    all_positions = []
    
    with open(pgn_path, "r", encoding="latin-1", errors="ignore") as f:
        pgn_text = f.read()
    
    pgn_stream = io.StringIO(pgn_text)
    filename = os.path.basename(pgn_path)
    #pbar = tqdm(desc=f"Lecture {filename}", unit=" parties", position=None)
    
    for game in iter(lambda: chess.pgn.read_game(pgn_stream), None):
        #pbar.update(1)
        if not is_game_valid(game):
            continue
        all_positions.extend(extract_positions_from_game(game))
    
    #pbar.close()
    
    if not all_positions:
        return np.empty((0, 72), dtype=np.float32)
    
    return np.stack(all_positions, axis=0).astype(np.float32)

def process_single_file(args):
    """Fonction wrapper pour traiter un fichier dans un processus séparé."""
    idx, total, pgn_file = args
    filename = os.path.basename(pgn_file)
    
    try:
        positions = extract_positions_from_pgn_file(pgn_file)
        print(f"[{idx}/{total}] {filename}: ✓ {len(positions)} positions extraites")
        return positions
    except Exception as e:
        print(f"[{idx}/{total}] {filename}: ✗ Erreur: {str(e)}")
        return np.empty((0, 72), dtype=np.float32)

def main():
    pgn_dir = "pgn_files"
    output_dir = "output_chunks"
    num_processes = 6  # Nombre de cœurs à utiliser
    chunk_size = 10_000_000  # Sauvegarder tous les 10 millions de positions
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Récupérer tous les fichiers twic*.pgn et les trier numériquement
    pgn_pattern = os.path.join(pgn_dir, "twic*.pgn")
    pgn_files = sorted(glob.glob(pgn_pattern), 
                       key=lambda x: int(x.split('twic')[1].split('.')[0]))
    
    if not pgn_files:
        print(f"Erreur: Aucun fichier twic*.pgn trouvé dans '{pgn_dir}'")
        return
    
    print(f"Trouvé {len(pgn_files)} fichiers PGN à traiter")
    print(f"Premier fichier: {os.path.basename(pgn_files[0])}")
    print(f"Dernier fichier: {os.path.basename(pgn_files[-1])}")
    print(f"Utilisation de {num_processes} processus parallèles")
    print(f"Sauvegarde tous les {chunk_size:,} positions")
    print()
    
    # Préparer les arguments pour le pool
    args_list = [(idx + 1, len(pgn_files), pgn_file) 
                 for idx, pgn_file in enumerate(pgn_files)]
    
    # Variables pour gérer les chunks
    all_positions = []
    total_positions_saved = 0
    chunk_number = 0
    buffer = None  # Buffer pour stocker l'excédent de positions
    
    # Traiter les fichiers en parallèle
    print("Début du traitement parallèle...\n")
    with Pool(processes=num_processes) as pool:
        for positions in pool.imap_unordered(process_single_file, args_list):
            # Ignorer les tableaux vides
            if len(positions) == 0:
                continue
            
            all_positions.append(positions)
            current_count = sum(len(p) for p in all_positions)
            
            # Sauvegarder si on dépasse le seuil
            while current_count >= chunk_size:
                chunk_number += 1
                
                # Concaténer toutes les positions accumulées
                combined = np.vstack(all_positions)
                
                # Extraire exactement chunk_size positions
                chunk_positions = combined[:chunk_size]
                remaining = combined[chunk_size:]
                
                chunk_path = os.path.join(output_dir, f"positions_chunk_{chunk_number:03d}.npy")
                
                print(f"\n💾 Sauvegarde du chunk {chunk_number}: {len(chunk_positions):,} positions")
                np.save(chunk_path, chunk_positions)
                
                total_positions_saved += len(chunk_positions)
                file_size = os.path.getsize(chunk_path) / (1024**2)  # MB
                print(f"   Fichier créé: {os.path.basename(chunk_path)} ({file_size:.1f} MB)")
                print(f"   Total sauvegardé: {total_positions_saved:,} positions\n")
                
                # Garder les positions restantes pour le prochain chunk
                if len(remaining) > 0:
                    all_positions = [remaining]
                    current_count = len(remaining)
                else:
                    all_positions = []
                    current_count = 0
    
    # Sauvegarder les positions restantes
    if all_positions:
        chunk_number += 1
        chunk_positions = np.vstack(all_positions)
        chunk_path = os.path.join(output_dir, f"positions_chunk_{chunk_number:03d}.npy")
        
        print(f"\n💾 Sauvegarde du chunk final {chunk_number}: {len(chunk_positions):,} positions")
        np.save(chunk_path, chunk_positions)
        
        total_positions_saved += len(chunk_positions)
        file_size = os.path.getsize(chunk_path) / (1024**2)  # MB
        print(f"   Fichier créé: {os.path.basename(chunk_path)} ({file_size:.1f} MB)")
    
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"{'='*60}")
    print(f"Fichiers PGN traités: {len(pgn_files)}")
    print(f"Chunks créés: {chunk_number}")
    print(f"Total de positions: {total_positions_saved:,}")
    print(f"Structure: [64 positions pièces, tour, règle 50 coups, 4 droits de roque, répétitions, gagnant]")
    print(f"Dossier de sortie: {output_dir}/")
    print("✓ Terminé !")


if __name__ == "__main__":
    main()