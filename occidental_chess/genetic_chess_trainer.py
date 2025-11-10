import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ai_pre_training.model import ChessEvaluationNet
from AI_vs_AI_tournament import play_tournament

class GeneticTrainer:
    """
    Entraîneur basé sur un algorithme génétique pour l'IA d'échecs
    """
    def __init__(
        self,
        base_model_path,
        population_size=20,
        n_opponents=10,
        elite_size=4,
        mutation_rate=0.15,
        mutation_sigma=0.01,
        crossover_rate=0.7,
        save_dir="genetic_training",
        n_workers=10
    ):
        """
        Args:
            base_model_path: Chemin vers le modèle pré-entraîné
            population_size: Taille de la population
            n_opponents: Nombre d'adversaires que chaque IA affronte
            elite_size: Nombre d'individus élites à conserver
            mutation_rate: Probabilité de mutation pour chaque poids
            mutation_sigma: Écart-type du bruit gaussien pour les mutations
            crossover_rate: Probabilité de crossover
            save_dir: Dossier pour sauvegarder les checkpoints
            n_workers: Nombre de processus parallèles (6 recommandé)
        """
        self.base_model_path = base_model_path
        self.population_size = population_size
        self.n_opponents = n_opponents
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.save_dir = Path(save_dir)
        self.n_workers = n_workers
        
        # Créer les dossiers nécessaires
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir = self.save_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # État de l'entraînement
        self.current_generation = 0
        self.population = []
        self.fitness_history = []
        self.best_individual_history = []
        
        # Fichiers de suivi
        self.log_file = self.logs_dir / "training_log.txt"
        self.metrics_file = self.logs_dir / "metrics.json"
        self.progress_file = self.save_dir / "progress.txt"
        
    def initialize_population(self):
        """Crée la population initiale à partir du modèle pré-entraîné"""
        print(f"\n{'='*60}")
        print(f"INITIALISATION DE LA POPULATION")
        print(f"{'='*60}")
        
        # Charger le modèle de base
        device = torch.device("cpu")
        base_model = ChessEvaluationNet()
        
        # Charger le checkpoint complet
        checkpoint = torch.load(self.base_model_path, map_location=device)
        
        # Extraire uniquement les poids du modèle
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Si c'est directement un state_dict
            base_model.load_state_dict(checkpoint)
        
        self.population = []
        
        for i in range(self.population_size):
            # Créer un clone du modèle
            individual = ChessEvaluationNet()
            individual.load_state_dict(copy.deepcopy(base_model.state_dict()))
            
            # Appliquer une mutation initiale légère pour créer de la diversité
            if i > 0:  # Le premier individu reste identique au modèle de base
                self._mutate(individual, sigma=self.mutation_sigma * 0.5)
            
            self.population.append({
                'model': individual,
                'fitness': 0.0,
                'id': f"gen{self.current_generation}_ind{i}"
            })
            
            print(f"Individu {i+1}/{self.population_size} créé")
        
        print(f"\nPopulation de {self.population_size} individus initialisée")
        self._log(f"Population initialisée avec {self.population_size} individus")
    
    def _mutate(self, model, sigma=None):
        """Applique une mutation gaussienne aux poids du modèle"""
        if sigma is None:
            sigma = self.mutation_sigma
            
        with torch.no_grad():
            for param in model.parameters():
                # Masque pour sélectionner les poids à muter
                mask = torch.rand_like(param) < self.mutation_rate
                # Bruit gaussien
                noise = torch.randn_like(param) * sigma
                # Appliquer la mutation
                param.data += mask.float() * noise
    
    def _crossover(self, parent1, parent2):
        """Crée un enfant par crossover uniforme de deux parents"""
        child = ChessEvaluationNet()
        
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(
                child.parameters(),
                parent1.parameters(),
                parent2.parameters()
            ):
                # Masque aléatoire pour choisir parent1 ou parent2
                mask = torch.rand_like(child_param) < 0.5
                child_param.data = torch.where(mask, p1_param.data, p2_param.data)
        
        return child
    
    def _tournament_selection(self, fitness_scores, k=3):
        """Sélection par tournoi : choisit le meilleur parmi k individus aléatoires"""
        candidates = np.random.choice(len(fitness_scores), k, replace=False)
        best_idx = candidates[np.argmax([fitness_scores[i] for i in candidates])]
        return best_idx
    
    def _save_model(self, model, path, simple=False):
        """Sauvegarde un modèle"""
        if simple:
            # Sauvegarde simple (juste les poids)
            torch.save(model.state_dict(), path)
        else:
            # Sauvegarde avec métadonnées (compatibilité checkpoint)
            torch.save({
                'model_state_dict': model.state_dict()
            }, path)
    
    def _load_model(self, path, simple=False):
        """Charge un modèle"""
        model = ChessEvaluationNet()
        
        # Charger le checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        if simple:
            # Chargement simple (juste les poids)
            model.load_state_dict(checkpoint)
        else:
            # Extraire les poids du modèle depuis le checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        return model
    
    def evaluate_fitness(self):
        """Évalue la fitness de toute la population via tournoi round-robin partiel"""
        print(f"\n{'='*60}")
        print(f"ÉVALUATION DE LA FITNESS - GÉNÉRATION {self.current_generation}")
        print(f"{'='*60}")
        
        # Sauvegarder temporairement tous les modèles
        temp_dir = self.save_dir / "temp_models"
        temp_dir.mkdir(exist_ok=True)
        
        model_paths = []
        for i, individual in enumerate(self.population):
            path = temp_dir / f"temp_model_{i}.pth"
            # Sauvegarder directement au format attendu par play_tournament
            torch.save({'model_state_dict': individual['model'].state_dict()}, path)
            model_paths.append(path)
            individual['fitness'] = 0.0
        
        # Créer les matchups : chaque individu affronte n_opponents adversaires
        matchups = []
        for i in range(self.population_size):
            # Choisir n_opponents adversaires aléatoires (différents de i)
            opponents = np.random.choice(
                [j for j in range(self.population_size) if j != i],
                size=min(self.n_opponents, self.population_size - 1),
                replace=False
            )
            for opponent in opponents:
                if i < opponent:  # Éviter les doublons (i vs j et j vs i)
                    matchups.append((i, opponent))
        
        total_matchups = len(matchups)
        print(f"\nTotal de confrontations : {total_matchups}")
        print(f"Parties par confrontation : 8")
        print(f"Total de parties à jouer : {total_matchups * 8}")
        print(f"Processus parallèles : {self.n_workers}")
        
        # Exécuter les matchups en parallèle
        completed = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Soumettre tous les matchups
            future_to_matchup = {}
            for idx1, idx2 in matchups:
                future = executor.submit(
                    self._run_matchup_wrapper,
                    str(model_paths[idx1]),
                    str(model_paths[idx2]),
                    f"AI_{idx1}",
                    f"AI_{idx2}"
                )
                future_to_matchup[future] = (idx1, idx2)
            
            # Récupérer les résultats au fur et à mesure
            for future in as_completed(future_to_matchup):
                idx1, idx2 = future_to_matchup[future]
                try:
                    results = future.result()
                    
                    # Mettre à jour les fitness
                    # Victoire = 1 point, Nul = 0.5 points
                    ai1_name = f"AI_{idx1}"
                    ai2_name = f"AI_{idx2}"
                    
                    self.population[idx1]['fitness'] += results[ai1_name] * 3
                    self.population[idx2]['fitness'] += results[ai2_name] * 3
                    self.population[idx1]['fitness'] += results['Nuls'] * 1
                    self.population[idx2]['fitness'] += results['Nuls'] * 1
                    
                    completed += 1
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed) * (total_matchups - completed)
                    
                    print(f"\rProgrès: {completed}/{total_matchups} "
                          f"({completed/total_matchups*100:.1f}%) - "
                          f"ETA: {eta/60:.1f} min", end='', flush=True)
                    
                    # Mise à jour du fichier de progression
                    self._update_progress(completed, total_matchups)
                    
                except Exception as e:
                    print(f"\nErreur lors du matchup {idx1} vs {idx2}: {e}")
        
        print(f"\n\nÉvaluation terminée en {(time.time()-start_time)/60:.1f} minutes")
        
        # Nettoyer les fichiers temporaires
        for path in model_paths:
            if path.exists():
                path.unlink()
        temp_dir.rmdir()
        
        # Normaliser les fitness (optionnel)
        fitness_scores = [ind['fitness'] for ind in self.population]
        print(f"\nFitness - Min: {min(fitness_scores):.2f}, "
              f"Max: {max(fitness_scores):.2f}, "
              f"Moyenne: {np.mean(fitness_scores):.2f}")
        
        return fitness_scores
    
    @staticmethod
    def _run_matchup_wrapper(path1, path2, name1, name2):
        """Wrapper pour exécuter un matchup dans un processus séparé"""
        from AI_vs_AI_tournament import play_tournament
        
        #print(f"DEBUG: Démarrage matchup {name1} vs {name2}", flush=True)
        
        # Les modèles sont déjà au bon format, on peut directement les utiliser
        results, _ = play_tournament(path1, path2, name1, name2)
        
        #print(f"DEBUG: Matchup terminé {name1} vs {name2}", flush=True)
        
        return results
    
    def create_next_generation(self, fitness_scores):
        """Crée la nouvelle génération via sélection, crossover et mutation"""
        print(f"\n{'='*60}")
        print(f"CRÉATION DE LA GÉNÉRATION {self.current_generation + 1}")
        print(f"{'='*60}")
        
        # Trier la population par fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        new_population = []
        
        # 1. ÉLITISME : Conserver les meilleurs individus
        print(f"\n1. Élitisme : conservation des {self.elite_size} meilleurs")
        for i in range(self.elite_size):
            idx = sorted_indices[i]
            elite = copy.deepcopy(self.population[idx])
            elite['id'] = f"gen{self.current_generation+1}_elite{i}"
            new_population.append(elite)
            print(f"   Élite {i+1} : fitness = {fitness_scores[idx]:.2f}")
        
        # 2. CROSSOVER + MUTATION pour remplir le reste
        print(f"\n2. Génération de {self.population_size - self.elite_size} nouveaux individus")
        
        while len(new_population) < self.population_size:
            # Sélection des parents par tournoi
            parent1_idx = self._tournament_selection(fitness_scores)
            parent2_idx = self._tournament_selection(fitness_scores)
            
            # Crossover
            if np.random.rand() < self.crossover_rate:
                child = self._crossover(
                    self.population[parent1_idx]['model'],
                    self.population[parent2_idx]['model']
                )
            else:
                # Pas de crossover : cloner un parent
                child = ChessEvaluationNet()
                child.load_state_dict(
                    copy.deepcopy(self.population[parent1_idx]['model'].state_dict())
                )
            
            # Mutation
            self._mutate(child)
            
            # Ajouter à la nouvelle population
            new_population.append({
                'model': child,
                'fitness': 0.0,
                'id': f"gen{self.current_generation+1}_ind{len(new_population)}"
            })
        
        self.population = new_population
        self.current_generation += 1
        
        print(f"\nNouvelle génération {self.current_generation} créée")
    
    def save_checkpoint(self, fitness_scores):
        """Sauvegarde un checkpoint complet de la génération actuelle"""
        checkpoint_dir = self.checkpoints_dir / f"generation_{self.current_generation}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Sauvegarder tous les modèles
        for i, individual in enumerate(self.population):
            model_path = checkpoint_dir / f"individual_{i}.pth"
            self._save_model(individual['model'], model_path)
        
        # Sauvegarder les métadonnées
        metadata = {
            'generation': self.current_generation,
            'population_size': self.population_size,
            'fitness_scores': fitness_scores,
            'best_fitness': max(fitness_scores),
            'avg_fitness': float(np.mean(fitness_scores)),
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'mutation_rate': self.mutation_rate,
                'mutation_sigma': self.mutation_sigma,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size,
                'n_opponents': self.n_opponents
            }
        }
        
        with open(checkpoint_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Sauvegarder le meilleur modèle séparément
        best_idx = np.argmax(fitness_scores)
        best_model_path = self.save_dir / f"best_model_gen{self.current_generation}.pth"
        self._save_model(self.population[best_idx]['model'], best_model_path)
        
        # Mettre à jour le fichier des métriques
        self.fitness_history.append({
            'generation': self.current_generation,
            'best': max(fitness_scores),
            'avg': float(np.mean(fitness_scores)),
            'min': min(fitness_scores),
            'std': float(np.std(fitness_scores))
        })
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.fitness_history, f, indent=2)
        
        print(f"\nCheckpoint sauvegardé : {checkpoint_dir}")
        self._log(f"Génération {self.current_generation} sauvegardée - "
                  f"Best fitness: {max(fitness_scores):.2f}")
    
    def load_checkpoint(self, generation=None):
        """Charge un checkpoint pour reprendre l'entraînement"""
        if generation is None:
            # Trouver la dernière génération
            checkpoint_dirs = list(self.checkpoints_dir.glob("generation_*"))
            if not checkpoint_dirs:
                print("Aucun checkpoint trouvé")
                return False
            
            generations = [int(d.name.split('_')[1]) for d in checkpoint_dirs]
            generation = max(generations)
        
        checkpoint_dir = self.checkpoints_dir / f"generation_{generation}"
        
        if not checkpoint_dir.exists():
            print(f"Checkpoint génération {generation} introuvable")
            return False
        
        print(f"\n{'='*60}")
        print(f"CHARGEMENT DU CHECKPOINT - GÉNÉRATION {generation}")
        print(f"{'='*60}")
        
        # Charger les métadonnées
        with open(checkpoint_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Charger les modèles
        self.population = []
        
        for i in range(self.population_size):
            model_path = checkpoint_dir / f"individual_{i}.pth"
            if model_path.exists():
                model = self._load_model(model_path)
                self.population.append({
                    'model': model,
                    'fitness': metadata['fitness_scores'][i],
                    'id': f"gen{generation}_ind{i}"
                })
        
        self.current_generation = generation
        
        # Charger l'historique des métriques
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.fitness_history = json.load(f)
        
        print(f"Checkpoint chargé : génération {generation}")
        print(f"Meilleure fitness : {metadata['best_fitness']:.2f}")
        print(f"Fitness moyenne : {metadata['avg_fitness']:.2f}")
        
        self._log(f"Reprise de l'entraînement depuis la génération {generation}")
        
        return True
    
    def _log(self, message):
        """Écrit dans le fichier de log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message)
    
    def _update_progress(self, current, total):
        """Met à jour le fichier de progression"""
        with open(self.progress_file, 'w') as f:
            f.write(f"Génération: {self.current_generation}\n")
            f.write(f"Progression: {current}/{total} ({current/total*100:.1f}%)\n")
            f.write(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def print_summary(self, fitness_scores):
        """Affiche un résumé de la génération actuelle"""
        print(f"\n{'='*60}")
        print(f"RÉSUMÉ GÉNÉRATION {self.current_generation}")
        print(f"{'='*60}")
        
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        print(f"\nTop 5 individus:")
        for i in range(min(5, len(fitness_scores))):
            idx = sorted_indices[i]
            print(f"  {i+1}. Individu {idx} - Fitness: {fitness_scores[idx]:.2f}")
        
        print(f"\nStatistiques:")
        print(f"  Meilleure fitness: {max(fitness_scores):.2f}")
        print(f"  Fitness moyenne: {np.mean(fitness_scores):.2f}")
        print(f"  Fitness médiane: {np.median(fitness_scores):.2f}")
        print(f"  Écart-type: {np.std(fitness_scores):.2f}")
        print(f"  Diversité (variance): {np.var(fitness_scores):.2f}")
        
        # Historique
        if len(self.fitness_history) > 1:
            print(f"\nÉvolution:")
            prev = self.fitness_history[-2]
            curr = self.fitness_history[-1]
            improvement = curr['best'] - prev['best']
            print(f"  Amélioration best fitness: {improvement:+.2f}")
            print(f"  Amélioration avg fitness: {curr['avg'] - prev['avg']:+.2f}")
        
        print(f"{'='*60}\n")
    
    def train(self, n_generations, resume=True):
        """
        Lance l'entraînement pour n_generations
        
        Args:
            n_generations: Nombre de générations à entraîner
            resume: Si True, tente de reprendre depuis le dernier checkpoint
        """
        print(f"\n{'#'*60}")
        print(f"DÉMARRAGE DE L'ENTRAÎNEMENT GÉNÉTIQUE")
        print(f"{'#'*60}")
        print(f"Population: {self.population_size} individus")
        print(f"Adversaires par individu: {self.n_opponents}")
        print(f"Générations: {n_generations}")
        print(f"Processus parallèles: {self.n_workers}")
        print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        # Tentative de reprise
        if resume and self.load_checkpoint():
            print(f"\nReprise depuis la génération {self.current_generation}")
            start_gen = self.current_generation + 1
        else:
            # Initialisation
            self.initialize_population()
            start_gen = 0
        
        # Boucle d'entraînement
        for gen in range(start_gen, start_gen + n_generations):
            gen_start = time.time()
            
            print(f"\n\n{'#'*60}")
            print(f"GÉNÉRATION {gen}")
            print(f"{'#'*60}")
            
            # Évaluation
            fitness_scores = self.evaluate_fitness()
            
            # Résumé
            self.print_summary(fitness_scores)
            
            # Sauvegarde
            self.save_checkpoint(fitness_scores)
            
            # Vérifier critère d'arrêt (plateau)
            if self._check_plateau(threshold=20):
                print("\n⚠️  Plateau détecté : aucune amélioration depuis 20 générations")
                print("Arrêt de l'entraînement")
                break
            
            # Créer la prochaine génération (sauf pour la dernière itération)
            if gen < start_gen + n_generations - 1:
                self.create_next_generation(fitness_scores)
            
            gen_time = (time.time() - gen_start) / 60
            print(f"\nTemps pour cette génération: {gen_time:.1f} minutes")
        
        total_time = (time.time() - start_time) / 60
        print(f"\n{'#'*60}")
        print(f"ENTRAÎNEMENT TERMINÉ")
        print(f"{'#'*60}")
        print(f"Temps total: {total_time:.1f} minutes")
        print(f"Générations complétées: {self.current_generation}")
        print(f"Meilleur modèle: {self.save_dir}/best_model_gen{self.current_generation}.pth")
        print(f"{'#'*60}\n")
        
        self._log(f"Entraînement terminé - {self.current_generation} générations - "
                  f"{total_time:.1f} minutes")
    
    def _check_plateau(self, threshold=20):
        """Vérifie si on est sur un plateau (pas d'amélioration)"""
        if len(self.fitness_history) < threshold + 1:
            return False
        
        recent = self.fitness_history[-threshold:]
        best_fitnesses = [gen['best'] for gen in recent]
        
        # Si aucune amélioration significative
        if max(best_fitnesses) - min(best_fitnesses) < 0.1:
            return True
        
        return False


# UTILISATION
if __name__ == "__main__":
    # Configuration
    BASE_MODEL = "ai_pre_training/checkpoints/model_final_MSE_30M.pth"
    
    trainer = GeneticTrainer(
        base_model_path=BASE_MODEL,
        population_size=20,
        n_opponents=10,
        elite_size=4,
        mutation_rate=0.15,
        mutation_sigma=0.01,
        crossover_rate=0.7,
        save_dir="genetic_training",
        n_workers=10
    )
    
    # Lancer l'entraînement (reprend automatiquement si interrompu)
    trainer.train(n_generations=50, resume=True)
    
    # Pour reprendre manuellement depuis une génération spécifique:
    # trainer.load_checkpoint(generation=10)
    # trainer.train(n_generations=40, resume=False)