import chess
import torch
import sys
import os
from ai_pre_training.model import ChessEvaluationNet
from ai_pre_training.dataset.extract_positions_from_pgn import board_to_numpy
from ai_pre_training.load_data import ChessDataset2
from torch.utils.data import DataLoader
import random
import numpy as np


def repetition_count(board: chess.Board) -> int:
    """
    Retourne le nombre de fois que la position actuelle du board
    (selon les règles de répétition de python-chess) est déjà apparue
    dans l'historique de la partie.
    """
    # Compter les occurrences de chaque position dans l'historique
    seen_positions = {}
    temp_board = chess.Board()
    
    # Parcourir tout l'historique des coups
    for move in board.move_stack:
        # Enregistrer la position avant de jouer le coup
        key = temp_board._transposition_key()
        seen_positions[key] = seen_positions.get(key, 0) + 1
        # Jouer le coup
        temp_board.push(move)
    
    # Compter la position actuelle (après tous les coups)
    current_key = temp_board._transposition_key()
    count = seen_positions.get(current_key, 0) + 1
    
    return count


def load_model(model_path, device):
    """Charge un modèle d'évaluation d'échecs depuis un checkpoint."""
    model = ChessEvaluationNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def find_checkmate_move(board, legal_moves):
    """Vérifie si un coup mène directement au mat."""
    for move in legal_moves:
        board_temp = board.copy()
        board_temp.push(move)
        if board_temp.is_checkmate():
            return move
    return None


def evaluate_positions(board_list, model, device):
    """
    Évalue une liste de positions avec le modèle.
    
    Args:
        board_list: Liste de boards à évaluer
        model: Modèle d'évaluation
        device: Device PyTorch
    
    Returns:
        Liste des évaluations (numpy array)
    """
    positions = []
    for board_temp in board_list:
        #print("board_temp:", board_temp)
        rep = repetition_count(board_temp)
        positions.append(board_to_numpy(board_temp, rep))
    
    positions_loader = DataLoader(
        ChessDataset2(positions),
        batch_size=len(positions),
        shuffle=False,
        num_workers=1
    )
    
    all_outputs = []
    with torch.no_grad():
        for position_batch in positions_loader:
            position_batch = position_batch.to(device)
            outputs = model(position_batch)
            all_outputs.extend(outputs.cpu().numpy())
    return np.array(all_outputs)


def minimax_search(board, model, device, depth, top_n, initial_depth, alpha=-float('inf'), beta=float('inf'), is_maximizing=None):
    """
    Recherche minimax optimisée avec élagage alpha-beta.
    """
    #print(f"\n\n\nMinimax depth {depth} called. Board:\n{board}\n")
    if is_maximizing is None:
        is_maximizing = (board.turn == chess.WHITE)
    
    # Cas terminal: profondeur 0 ou partie terminée
    if depth == 0 or board.is_game_over():
        # Évaluer la position
        rep = repetition_count(board)
        position = board_to_numpy(board, rep)
        dataset = ChessDataset2([position])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                eval_score = model(batch).cpu().item()
                #print(f"Depth 0 or game over reached at depth {initial_depth}. Evaluated score: {eval_score}.")
                return eval_score
    
    # Obtenir les coups légaux
    legal_moves = list(board.legal_moves)
    
    # AJOUT : Vérifier s'il y a des coups légaux
    if not legal_moves:
        # Pas de coups légaux = mat ou pat
        if board.is_checkmate():
            mate_score = -float('inf') if is_maximizing else float('inf')
        else:
            # Pat ou autre = nulle (score 0)
            mate_score = 0
        
        if depth == initial_depth:
            return (None, mate_score)
        else:
            return mate_score
    
    # Évaluer tous les coups légaux
    candidate_boards = []
    for move in legal_moves:
        board_temp = board.copy()
        board_temp.push(move)
        #print(f"Evaluating move: {move}, resulting board:\n{board_temp}")
        if board_temp.is_checkmate():
            mate_score = float('inf') if is_maximizing else -float('inf')
            if depth == initial_depth:
                return (move, mate_score)
            else:
                return mate_score
        candidate_boards.append(board_temp)
    
    evaluations = evaluate_positions(candidate_boards, model, device).squeeze()
    #print(f"On veut {'maximiser' if is_maximizing else 'minimiser'}")
    #print("Liste des évaluations des coups légaux:", evaluations)

    for indice, move in enumerate(legal_moves):
        board_temp = board.copy()
        board_temp.push(move)
        if isinstance(evaluations.tolist(), (list, tuple)):
            val = evaluations.tolist()[indice]
        else:
            # Sinon, c'est juste un float ou une valeur scalaire
            val = evaluations.tolist()
        #print(f"Move: {move}, Evaluation: {val}\n{board_temp}")
    # Trier et sélectionner les top_n meilleurs coups
    if is_maximizing:
        top_indices = np.argsort(evaluations)[-top_n:][::-1]
    else:
        top_indices = np.argsort(evaluations)[:top_n]
    
    # Initialiser la meilleure valeur
    best_value = -float('inf') if is_maximizing else float('inf')
    best_move = None

    # Explorer chaque coup candidat
    for idx in top_indices:
        move = legal_moves[int(idx)]
        board_temp = board.copy()
        board_temp.push(move)
        if isinstance(evaluations.tolist(), (list, tuple)):
            val = evaluations.tolist()[idx]
        else:
            # Sinon, c'est juste un float ou une valeur scalaire
            val = evaluations.tolist()
        #print(f"Coup candidat: {move} at depth {depth} with eval {val}")
        
    # Explorer chaque coup candidat
    for idx in top_indices:
        move = legal_moves[int(idx)]
        board_temp = board.copy()
        board_temp.push(move)
        #print(f"Coup candidat: {move} at depth {depth}")
        
        # Appel récursif - retourne toujours un float (pas un tuple)
        eval_score = minimax_search(
            board_temp, 
            model, 
            device, 
            depth - 1, 
            top_n,
            initial_depth,
            alpha, 
            beta, 
            not is_maximizing
        )
        
        # Mise à jour de la meilleure valeur
        if is_maximizing:
            if eval_score > best_value:
                best_value = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
        else:
            if eval_score < best_value:
                best_value = eval_score
                best_move = move
            beta = min(beta, eval_score)
        
        # Élagage alpha-beta
        if beta <= alpha:
            break
    
    # AJOUT : Vérification de sécurité
    if best_move is None and legal_moves:
        # Si aucun coup n'a été trouvé (ne devrait pas arriver), prendre le premier coup légal
        best_move = legal_moves[0]
        #print(f"WARNING: Aucun meilleur coup trouvé, utilisation du premier coup légal: {best_move}")
    
    #print(f"Fin Minimax, depth {initial_depth}: Meilleur coup {best_move}, Éval {best_value}")
    if depth == initial_depth:
        return (best_move, best_value)
    else:
        return best_value


def select_best_move(board, legal_moves, model, device, use_deep_search=True):
    """
    Sélectionne le meilleur coup selon l'évaluation du modèle.
    
    Args:
        use_deep_search: Si True, utilise la recherche profonde (3 demi-coups)
    """
    #print(f"Selecting best move. Deep search: {use_deep_search}")
    if use_deep_search:
        return minimax_search(board, model, device, depth=3, top_n=3, initial_depth=3)
    else:
        # Ancienne méthode (1 coup à l'avance)
        checkmate_move = find_checkmate_move(board, legal_moves)
        if checkmate_move is not None:
            return checkmate_move, None
        
        positions = []
        for move in legal_moves:
            board_temp = board.copy()
            board_temp.push(move)
            rep = repetition_count(board_temp)
            positions.append(board_to_numpy(board_temp, rep))
        
        positions_loader = DataLoader(
            ChessDataset2(positions),
            batch_size=len(positions),
            shuffle=False,
            num_workers=1
        )
        
        all_outputs = []
        with torch.no_grad():
            for position_batch in positions_loader:
                position_batch = position_batch.to(device)
                outputs = model(position_batch)
                all_outputs.extend(outputs.cpu().numpy())
        
        if board.turn == chess.WHITE:
            best_move_idx = np.argmax(all_outputs)
        else:
            best_move_idx = np.argmin(all_outputs)
        
        return legal_moves[best_move_idx], all_outputs[best_move_idx]


def get_game_outcome(board):
    """
    Détermine la raison de la fin de partie.
    
    Returns:
        tuple: (résultat, raison de la fin)
    """
    result = board.result()
    
    if board.is_checkmate():
        winner = "Blancs" if board.turn == chess.BLACK else "Noirs"
        reason = f"Mat - Victoire des {winner}"
    elif board.is_stalemate():
        reason = "Pat - Match nul"
    elif board.is_insufficient_material():
        reason = "Matériel insuffisant - Match nul"
    elif board.can_claim_threefold_repetition():
        reason = "Triple répétition - Match nul"
    elif board.is_seventyfive_moves():
        reason = "Règle des 75 coups - Match nul"
    elif board.is_fivefold_repetition():
        reason = "Répétition 5 fois - Match nul"
    elif board.can_claim_draw():
        if board.can_claim_fifty_moves():
            reason = "Règle des 50 coups possible - Match nul"
        else:
            reason = "Nulle réclamable - Match nul"
    else:
        reason = "Fin de partie (autre raison)"
    
    return result, reason


def play_ai_vs_ai(model_white, model_black, device, white_name, black_name, opening_move=None):
    """
    Fait jouer deux IA l'une contre l'autre.
    
    Args:
        model_white: Modèle pour les blancs
        model_black: Modèle pour les noirs
        device: Device PyTorch
        white_name: Nom de l'IA jouant les blancs
        black_name: Nom de l'IA jouant les noirs
        opening_move: Premier coup forcé pour les blancs (optionnel)
    
    Returns:
        dict: Statistiques de la partie
    """
    board = chess.Board()
    move_number = 0
    
    print("\n" + "="*50)
    print(f"DÉBUT DE LA PARTIE")
    print(f"Blancs: {white_name} | Noirs: {black_name}")
    if opening_move:
        print(f"Ouverture forcée: {opening_move}")
    print("="*50 + "\n")
    
    # Jouer le premier coup forcé si spécifié
    if opening_move:
        move = chess.Move.from_uci(opening_move)
        if move in board.legal_moves:
            board.push(move)
            move_number += 1
            print(f"Move {move_number} - Blancs ({white_name}): {move} (ouverture forcée)")
        else:
            print(f"ERREUR: Coup d'ouverture {opening_move} invalide!")
            return None
    
    while not board.is_game_over():
        # Vérifier la triple répétition avant de continuer
        if board.can_claim_threefold_repetition():
            break
        
        move_number += 1
        legal_moves = list(board.legal_moves)
        
        if board.turn == chess.WHITE:
            #print(f"Évaluation de la position pour les Blancs ({white_name})...")
            move, evaluation = select_best_move(board, legal_moves, model_white, device)
            if move is None:
                print("ERREUR: Aucun coup valide trouvé pour les Blancs!")
                break
            eval_str = f"(éval: {evaluation:.2f})" if evaluation is not None else "(mat forcé)"
            #print(f"Move {move_number} - Blancs ({white_name}): {move} {eval_str}")
        else:
            #print(f"Évaluation de la position pour les Noirs ({black_name})...")
            move, evaluation = select_best_move(board, legal_moves, model_black, device)
            if move is None:
                #print("ERREUR: Aucun coup valide trouvé pour les Noirs!")
                break
            eval_str = f"(éval: {evaluation:.2f})" if evaluation is not None else "(mat forcé)"
            #print(f"Move {move_number} - Noirs ({black_name}): {move} {eval_str}")

        board.push(move)
        
        #if move_number % 10 == 0:
        #print(str(board) + "\n")
    
    # Obtenir le résultat et la raison
    result, reason = get_game_outcome(board)
    total_moves = len(board.move_stack)
    total_plies = board.ply()  # Nombre de demi-coups
    
    print("\n" + "="*50)
    print("STATISTIQUES DE LA PARTIE")
    print("="*50)
    print(f"Nombre total de coups: {total_moves}")
    print(f"Nombre de demi-coups (plies): {total_plies}")
    print(f"Résultat: {result}")
    print(f"Raison: {reason}")
    
    if result == "1-0":
        print(f"Vainqueur: {white_name} (Blancs)")
    elif result == "0-1":
        print(f"Vainqueur: {black_name} (Noirs)")
    else:
        print("Match nul")
    
    print("="*50)
    print("\nPOSITION FINALE")
    print("="*50)
    print(board)
    print("="*50 + "\n")
    
    # Retourner les statistiques
    stats = {
        "result": result,
        "reason": reason,
        "total_moves": total_moves,
        "total_plies": total_plies,
        "white_name": white_name,
        "black_name": black_name
    }
    
    return stats


def play_tournament(model_path_1, model_path_2, model_name_1="MSE", model_name_2="L1"):
    """
    Fait jouer un tournoi entre deux IA avec 10 parties:
    - 5 parties avec chaque IA jouant les blancs
    - 5 ouvertures différentes: e2e4, d2d4, c2c4, g1f3, g2g3
    
    Args:
        model_path_1: Chemin vers les poids de l'IA 1
        model_path_2: Chemin vers les poids de l'IA 2
        model_name_1: Nom de l'IA 1
        model_name_2: Nom de l'IA 2
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")
    
    print(f"\nChargement du modèle 1 ({model_name_1}) : {model_path_1}")
    model_1 = load_model(model_path_1, device)
    
    print(f"Chargement du modèle 2 ({model_name_2}) : {model_path_2}")
    model_2 = load_model(model_path_2, device)
    
    # Les 5 ouvertures
    openings = ["e2e4", "d2d4", "c2c4", "g1f3", "g2g3"]
    opening_names = ["1. e4", "1. d4", "1. c4", "1. Nf3", "1. g3"]
    #openings = ["g1f3"]
    #opening_names = ["1. Nf3"]
    
    results = {
        model_name_1: 0,
        model_name_2: 0,
        "Nuls": 0
    }
    
    game_details = []
    total_moves = 0
    draw_reasons = {}
    
    # 10 parties au total
    for i in range(10):
        game_num = i + 1
        opening_idx = i % 5
        opening = openings[opening_idx]
        opening_name = opening_names[opening_idx]
        
        print(f"\n\n\n\n{'#'*60}")
        print(f"PARTIE {game_num}/10")
        print(f"Ouverture: {opening_name}")
        
        # Les 5 premières parties: IA 1 joue les blancs
        # Les 5 dernières parties: IA 2 joue les blancs
        if i < 5:
            white_model = model_1
            black_model = model_2
            white_name = model_name_1
            black_name = model_name_2
        else:
            white_model = model_2
            black_model = model_1
            white_name = model_name_2
            black_name = model_name_1
        
        print(f"{'#'*60}\n")
        
        stats = play_ai_vs_ai(white_model, black_model, device, white_name, black_name, opening)
        
        if stats is None:
            continue
        
        result = stats["result"]
        total_moves += stats["total_moves"]
        
        # Comptabiliser le résultat
        if result == "1-0":
            results[white_name] += 1
            winner = white_name
        elif result == "0-1":
            results[black_name] += 1
            winner = black_name
        else:
            results["Nuls"] += 1
            winner = "Nul"
            # Compter les raisons des nulles
            draw_reasons[stats["reason"]] = draw_reasons.get(stats["reason"], 0) + 1
        
        game_details.append({
            "game": game_num,
            "opening": opening_name,
            "white": white_name,
            "black": black_name,
            "result": result,
            "winner": winner,
            "moves": stats["total_moves"],
            "reason": stats["reason"]
        })
        
        print(f"\n{'='*60}")
        print(f"Score après {game_num} partie(s):")
        print(f"  {model_name_1}: {results[model_name_1]} victoires")
        print(f"  {model_name_2}: {results[model_name_2]} victoires")
        print(f"  Matchs nuls: {results['Nuls']}")
        print(f"{'='*60}")
    
    # Statistiques finales
    avg_moves = total_moves / 10
    
    print("\n" + "="*60)
    print("STATISTIQUES FINALES DU TOURNOI")
    print("="*60)
    print(f"\nTotal de parties : 10")
    print(f"{model_name_1} - Victoires : {results[model_name_1]} ({results[model_name_1]/10*100:.1f}%)")
    print(f"{model_name_2} - Victoires : {results[model_name_2]} ({results[model_name_2]/10*100:.1f}%)")
    print(f"Matchs nuls : {results['Nuls']} ({results['Nuls']/10*100:.1f}%)")
    print(f"\nNombre moyen de coups par partie : {avg_moves:.1f}")
    
    if draw_reasons:
        print("\nRaisons des matchs nuls :")
        for reason, count in draw_reasons.items():
            print(f"  - {reason}: {count}")
    
    print("\n" + "-"*60)
    print("DÉTAIL DES PARTIES")
    print("-"*60)
    for detail in game_details:
        print(f"Partie {detail['game']:2d} | {detail['opening']:8s} | "
              f"Blancs: {detail['white']:4s} vs Noirs: {detail['black']:4s} | "
              f"{detail['moves']:3d} coups | {detail['result']:7s} | {detail['winner']}")
        print(f"           Raison: {detail['reason']}")
    print("="*60)
    
    return results, game_details


# Exemple d'utilisation
if __name__ == "__main__":
    #MODEL_PATH_1 = "ai_pre_training/checkpoints/model_final_L1.pth"
    #MODEL_PATH_1 = "ai_pre_training/checkpoints/model_final_L1_30M.pth"
    MODEL_PATH_2 = "ai_pre_training/checkpoints/model_final_MSE.pth"
    MODEL_PATH_1 = "ai_pre_training/checkpoints/model_final_MSE_30M.pth"
    
    play_tournament(MODEL_PATH_1, MODEL_PATH_2, model_name_1="MSE_30M", model_name_2="MSE")