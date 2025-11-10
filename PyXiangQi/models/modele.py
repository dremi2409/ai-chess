from PyXiangQi.Gameplay.plt import Plateau, Couleur, PieceType
import random as rd
import time
import numpy as np

class Modele:
    #Interface graphique du XiangQi
    def __init__(self, model_name = None):
        self.name = model_name
        if self.name[:2]=="IA":
            #Charger le modèle IA
            self.modelllllle = None
            self.name = "IA"

    #Demande au modèle le prochain coup parmis ceux possibles
    def trouver_coup(self, plateau: Plateau):
        match self.name:
            #Joueur aléatoire
            case "Aleatoire":
                time.sleep(0.4)
                coup = rd.choice(plateau.get_coup_possible) 
                return int(coup[0]), int(coup[1]), int(coup[2]), int(coup[3])

            case "IA":
                input_tenseur = self.plt_to_tensor(plateau)
                print(input_tenseur)
                coup = plateau.get_coup_possible[0] 
                return int(coup[0]), int(coup[1]), int(coup[2]), int(coup[3])
          
            #Si modèle pas trouvé : on fait jouer l'humain
            case _:
                coup_legal = False
                while not(coup_legal):
                    try:
                        a = int(input("Ligne de la pièce à déplacer : "))
                        b = int(input("Colonne de la pièce à déplacer : "))
                        c = int(input("Ligne où deplacer la pièce : "))
                        d = int(input("Colonne où deplacer la pièce : "))

                        if plateau.grille_coups[a][b]:
                            if str(c)+str(d) in plateau.grille_coups[a][b]:
                                return a, b, c, d
                                    
                        print("Coup non légal")
                                    
                    except:
                        print("Coup non légal")

    def plt_to_tensor(self, plt: Plateau):
        #Transforme la position en grille 10*9*14 (Lignes - colonnes - pièces différentes)
        grille =plt.grille

        numpy_array_repet = np.zeros((plt.lignes,plt.colonnes,14+1))

        #Ajouter les pièces au tenseur
        for ligne in range(plt.lignes):
            for col in range(plt.colonnes):
                piece = grille[ligne][col]
                if piece != '-':
                    match piece.couleur:
                        case Couleur.RED:
                            coeff = 1

                        case Couleur.BLACK:
                            coeff = 0

                    match piece.type:
                        case PieceType.GENERAL:
                            base = 0
                        case PieceType.ASSISTANT:
                            base = 1
                        case PieceType.ELEPHANT:
                            base = 2
                        case PieceType.CANON:
                            base = 3
                        case PieceType.CHARIOT:
                            base = 4
                        case PieceType.CHEVAL:
                            base = 5
                        case PieceType.SOLDAT:
                            base = 6

                    numpy_array_repet[ligne][col][coeff*7+base] = 1

        #Ajouter le tour du joueur
        match plt.tour_actuel:
            case Couleur.RED:
                coeff = 1

            case Couleur.BLACK:
                coeff = 0

        numpy_array_repet[coeff][0][14] = 1 

        #Regarder les répetitions et le nombre de coups sans manger une pièce 
        numpy_array_repet[2][0][14] = plt.pas_de_prise
        numpy_array_repet[3][0][14] = plt.nombre_repet

        return numpy_array_repet


                        
