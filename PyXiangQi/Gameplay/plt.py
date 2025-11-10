from enum import Enum
import numpy as np
from typing import Optional
from itertools import chain
import copy

class Couleur(Enum):
    RED = "RED"
    BLACK = "BLACK"
    GREY = "GREY"

class PieceType(Enum):
    GENERAL = 1
    CHARIOT = 2
    CANON = 3
    CHEVAL = 4
    ELEPHANT = 5
    ASSISTANT = 6
    SOLDAT = 7

class Piece:
    """Représente une pièce d'échecs"""
    def __init__(self, type_piece: PieceType, couleur: Couleur):
        self.type = type_piece
        self.couleur = couleur
        self.a_bouge = False  # Pour le roque et la première avancée du pion
    
    def __repr__(self):
        return f"{self.type.value}"
    
    def symbole(self) -> str:
        """Retourne le symbole Unicode de la pièce"""
        symboles = {
            (PieceType.SOLDAT, Couleur.RED): "兵",
            (PieceType.ELEPHANT, Couleur.RED): "相",
            (PieceType.CANON, Couleur.RED): "炮",
            (PieceType.CHARIOT, Couleur.RED): "車",
            (PieceType.CHEVAL, Couleur.RED): "馬",
            (PieceType.GENERAL, Couleur.RED): "帥",
            (PieceType.ASSISTANT, Couleur.RED): "士",
            (PieceType.SOLDAT, Couleur.BLACK): "卒",
            (PieceType.ELEPHANT, Couleur.BLACK): "象",
            (PieceType.CANON, Couleur.BLACK): "砲",
            (PieceType.CHARIOT, Couleur.BLACK): "俥",
            (PieceType.CHEVAL, Couleur.BLACK): "傌",
            (PieceType.GENERAL, Couleur.BLACK): "將",
            (PieceType.ASSISTANT, Couleur.BLACK): "仕"
        }
        return symboles[(self.type, self.couleur)]

class Plateau:
    #Interface graphique du XiangQi
    def __init__(self, lignes, colonnes, plt = None, tour_actuel = Couleur.RED):
        self.lignes = lignes
        self.colonnes = colonnes
        self.victoire = None
        self.echec = None
        self.piece_echec = []
        self.repet = 1
        self.pas_de_prise = 0
        self.nombre_repet = 1

        if plt:
            self.grille=plt

        else:
            self.grille = np.array([[Piece(PieceType.CHARIOT, Couleur.BLACK),Piece(PieceType.CHEVAL, Couleur.BLACK),
                                 Piece(PieceType.ELEPHANT, Couleur.BLACK),Piece(PieceType.ASSISTANT, Couleur.BLACK),
                                 Piece(PieceType.GENERAL, Couleur.BLACK),
                                 Piece(PieceType.ASSISTANT, Couleur.BLACK),Piece(PieceType.ELEPHANT, Couleur.BLACK),
                                 Piece(PieceType.CHEVAL, Couleur.BLACK),Piece(PieceType.CHARIOT, Couleur.BLACK)],
                                ["-","-","-","-","-","-","-","-","-"],
                                ["-",Piece(PieceType.CANON, Couleur.BLACK),"-","-","-","-","-",
                                 Piece(PieceType.CANON, Couleur.BLACK),"-"],
                                [Piece(PieceType.SOLDAT, Couleur.BLACK),"-",Piece(PieceType.SOLDAT, Couleur.BLACK),"-",
                                 Piece(PieceType.SOLDAT, Couleur.BLACK),"-",Piece(PieceType.SOLDAT, Couleur.BLACK),"-",
                                 Piece(PieceType.SOLDAT, Couleur.BLACK)],
                                ["-","-","-","-","-","-","-","-","-"],
                                ["-","-","-","-","-","-","-","-","-"],
                                [Piece(PieceType.SOLDAT, Couleur.RED),"-",Piece(PieceType.SOLDAT, Couleur.RED),
                                 "-",Piece(PieceType.SOLDAT, Couleur.RED),"-",Piece(PieceType.SOLDAT, Couleur.RED),
                                 "-",Piece(PieceType.SOLDAT, Couleur.RED)],
                                ["-",Piece(PieceType.CANON, Couleur.RED),"-","-","-","-","-",
                                 Piece(PieceType.CANON, Couleur.RED),"-"],
                                ["-","-","-","-","-","-","-","-","-"],
                                [Piece(PieceType.CHARIOT, Couleur.RED),Piece(PieceType.CHEVAL, Couleur.RED),
                                 Piece(PieceType.ELEPHANT, Couleur.RED),Piece(PieceType.ASSISTANT, Couleur.RED),
                                 Piece(PieceType.GENERAL, Couleur.RED),
                                 Piece(PieceType.ASSISTANT, Couleur.RED),Piece(PieceType.ELEPHANT, Couleur.RED),
                                 Piece(PieceType.CHEVAL, Couleur.RED),Piece(PieceType.CHARIOT, Couleur.RED)]
                                ])
        
        self.tour_actuel = tour_actuel
        self.grille_coups = [[self.calcul_coups_legaux(self.grille, col, ligne) for col in range(self.colonnes)] for ligne in range(self.lignes)]
        self.get_coup_possible = self.all_get_coup_possible()
        
        self.historique = ["".join(list(map(lambda x: x if x=="-" else x.symbole(), chain.from_iterable(self.grille))))]

    def obtenir_piece_inverse(self, grille, col: int, ligne: int) -> Optional[Piece]:
        """Retourne la pièce à une position donnée"""
        if 0 <= ligne < self.lignes and 0 <= col < self.colonnes:
            return grille[self.lignes - 1 - ligne][self.colonnes - 1 - col]
        return None
    
    def obtenir_piece(self, grille, col: int, ligne: int) -> Optional[Piece]:
        """Retourne la pièce à une position donnée"""
        if 0 <= ligne < self.lignes and 0 <= col < self.colonnes:
            return grille[ligne][col]
        return None
    
    def changer_couleur(self):
        if self.tour_actuel == Couleur.RED:
            self.tour_actuel = Couleur.BLACK

        else:
            self.tour_actuel = Couleur.RED

    def deplacer_piece(self, init_col: int,init_ligne: int, col: int,ligne: int) -> None:
        self.grille[ligne][col] = self.grille[init_ligne][init_col] 
        self.grille[init_ligne][init_col] = "-"

        self.changer_couleur()
        self.grille_coups = [[self.calcul_coups_legaux(self.grille, col, ligne) for col in range(self.colonnes)] for ligne in range(self.lignes)]
        self.get_coup_possible = self.all_get_coup_possible()

        self.reajust_coup_legaux()
        self.get_coup_possible = self.all_get_coup_possible()

        self.coords_echec()

        if len(self.get_coup_possible) == 0 and self.tour_actuel == Couleur.BLACK:
            self.victoire = Couleur.RED
        if len(self.get_coup_possible) == 0 and self.tour_actuel == Couleur.RED:
            self.victoire = Couleur.BLACK

        #Créer une liste "id" pour l'historique, les pats, et le ML
        nouvelle_position =  "".join(list(map(lambda x: x if x=="-" else x.symbole(), chain.from_iterable(self.grille))))
        compteur_nulle=1
        for pos in self.historique:
            if pos == nouvelle_position:
                compteur_nulle+=1
        self.nombre_repet = compteur_nulle
        if self.nombre_repet == 4:
            self.victoire = Couleur.GREY

        self.repet = compteur_nulle + 1

        #Match nul si pas de prise au bout de 50 coup pour chaque joueur
        if self.historique[-1].count('-') == nouvelle_position.count('-'):
            self.pas_de_prise += 1
        else:
            self.pas_de_prise = 0
        if self.pas_de_prise == 100:
            self.victoire = Couleur.GREY

        self.historique.append(nouvelle_position)
    
    def calcul_coups_legaux(self, grille, col: int, ligne: int):
        piece = self.obtenir_piece(grille, col,ligne)
        if piece != "-":
            piece_info = piece.type
            piece_couleur = piece.couleur

            if piece_couleur == self.tour_actuel:
                match piece_info:
                    case PieceType.SOLDAT:
                        list_coup = self._coupsoldat(grille, piece_couleur, col, ligne)
                    case PieceType.ELEPHANT:
                        list_coup = self._coupelephant(grille, piece_couleur, col, ligne)
                    case PieceType.ASSISTANT:
                        list_coup = self._coupassistant(grille, piece_couleur, col, ligne) 
                    case PieceType.GENERAL:
                           list_coup = self._couproi(grille, piece_couleur, col, ligne) 
                    case PieceType.CHEVAL:
                        list_coup = self._coupcheval(grille, piece_couleur, col, ligne)
                    case PieceType.CHARIOT:
                        list_coup = self._couptour(grille, piece_couleur, col, ligne)
                    case PieceType.CANON:
                        list_coup = self._coupcanon(grille, piece_couleur, col, ligne)
                    
                    case _:
                        return None         
            else:
                return None    
        else:
            return None  
        if len(list_coup)==0:
            return None
        
        return list_coup
    
    def all_get_coup_possible(self):
        list_coups_possibles = []
        for ligne in range(self.lignes):
            for col in range(self.colonnes):
                if self.grille_coups[ligne][col]:
                    for coup in self.grille_coups[ligne][col]:
                        list_coups_possibles.append(str(ligne)+str(col)+coup)

        return list_coups_possibles
    
    def reajust_coup_legaux(self):
        #Regarder des potentiels echecs et mat infligés à soit-même
        for coup in self.get_coup_possible:
            coup_elimine = False

            self.grille_alternative = copy.deepcopy(self.grille)
            x_init=int(coup[1])
            y_init=int(coup[0])
            x=int(coup[3])
            y=int(coup[2])

            self.grille_alternative[y][x] = self.grille_alternative[y_init][x_init] 
            self.grille_alternative[y_init][x_init] = "-"
            localisation_roi = self.avoir_coords_roi(self.grille_alternative)

            self.changer_couleur()
            
            for ligne in range(self.lignes):
                for col in range(self.colonnes):
                    coups_possible = self.calcul_coups_legaux(self.grille_alternative, col,ligne)
                    if coups_possible:
                        if localisation_roi in coups_possible:  
                            coup_elimine = True    
                            self.grille_coups[y_init][x_init].remove(str(y)+str(x))
                            
                    if coup_elimine:
                        break

                if coup_elimine:
                    break

            self.changer_couleur()

    def coords_echec(self):
        self.piece_echec = []
        self.echec = None
        localisation_roi = self.avoir_coords_roi(self.grille)
        self.changer_couleur()

        for ligne in range(self.lignes):
            for col in range(self.colonnes):
                coups_possible = self.calcul_coups_legaux(self.grille, col,ligne)
                if coups_possible:
                    if localisation_roi in coups_possible:
                        if not(self.echec):
                            self.echec = localisation_roi
                            self.piece_echec = [str(ligne)+str(col)]

                        else:
                            self.piece_echec.append(str(ligne)+str(col))

        self.changer_couleur()

    def avoir_coords_roi(self, grille):
        if self.tour_actuel==Couleur.RED:  
            for ligne in range(7,10):
                for col in range(3,6):
                    piece = grille[ligne][col]
                    if piece != "-":
                        if piece.type == PieceType.GENERAL:
                            return str(ligne) + str(col)
        
        if self.tour_actuel==Couleur.BLACK:  
            for ligne in range(0,3):
                for col in range(3,6):
                    piece = grille[ligne][col]
                    if piece != "-":
                        if piece.type == PieceType.GENERAL:
                            return str(ligne) + str(col)
    
    def _coupsoldat(self, grille, piece_couleur, col, ligne):
        list_coup = []
        if piece_couleur == Couleur.RED:
            #Va tout droit si pas d'obstacle (piece de même couleur)
            if self.check_coup(grille, ligne-1,col,Couleur.RED, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-1)+str(col))
            #Va sur les côtés après la rivière
            if self.check_coup(grille, ligne,col+1,Couleur.RED, [0,5],[0,self.colonnes]):
                    list_coup.append(str(ligne)+str(col+1))
            if self.check_coup(grille, ligne,col-1,Couleur.RED, [0,5],[0,self.colonnes]):
                        list_coup.append(str(ligne)+str(col-1))

        if piece_couleur == Couleur.BLACK:
            if self.check_coup(grille, ligne+1,col,Couleur.BLACK, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+1)+str(col))
            if self.check_coup(grille, ligne,col+1,Couleur.BLACK, [5,10],[0,self.colonnes]):
                    list_coup.append(str(ligne)+str(col+1))
            if self.check_coup(grille, ligne,col-1,Couleur.BLACK, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne)+str(col-1))

        return list_coup
    
    def _coupelephant(self, grille, piece_couleur, col, ligne):
        list_coup = []
        if piece_couleur == Couleur.RED:
            #Va en diagonale de deux cases sauf si bloqué
            if self.check_coup(grille, ligne-1,col-1,Couleur.RED, [5,10],[0,self.colonnes],elephant_block=True) and self.check_coup(grille, ligne-2,col-2,Couleur.RED, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col-2))
            if self.check_coup(grille, ligne+1,col+1,Couleur.RED, [5,10],[0,self.colonnes],elephant_block=True) and self.check_coup(grille, ligne+2,col+2,Couleur.RED, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col+2))
            if self.check_coup(grille, ligne+1,col-1,Couleur.RED, [5,10],[0,self.colonnes],elephant_block=True) and self.check_coup(grille, ligne+2,col-2,Couleur.RED, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col-2))
            if self.check_coup(grille, ligne-1,col+1,Couleur.RED, [5,10],[0,self.colonnes],elephant_block=True) and self.check_coup(grille, ligne-2,col+2,Couleur.RED, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col+2))

        if piece_couleur == Couleur.BLACK:
            if self.check_coup(grille, ligne-1,col-1,Couleur.BLACK, [0,5], [0,self.colonnes], elephant_block=True) and self.check_coup(grille, ligne-2,col-2,Couleur.BLACK, [0,5], [0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col-2))
            if self.check_coup(grille, ligne+1,col+1,Couleur.BLACK, [0,5], [0,self.colonnes], elephant_block=True) and self.check_coup(grille, ligne+2,col+2,Couleur.BLACK, [0,5], [0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col+2))
            if self.check_coup(grille, ligne+1,col-1,Couleur.BLACK, [0,5], [0,self.colonnes], elephant_block=True) and self.check_coup(grille, ligne+2,col-2,Couleur.BLACK, [0,5], [0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col-2))
            if self.check_coup(grille, ligne-1,col+1,Couleur.BLACK, [0,5], [0,self.colonnes], elephant_block=True) and self.check_coup(grille, ligne-2,col+2,Couleur.BLACK, [0,5], [0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col+2))

        return list_coup
    
    def _coupassistant(self, grille, piece_couleur, col, ligne):
        list_coup = []
        if piece_couleur == Couleur.RED:
            #Va en diagonale dans un périmètre restreint
            if self.check_coup(grille, ligne-1,col-1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne-1)+str(col-1))
            if self.check_coup(grille, ligne+1,col+1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne+1)+str(col+1))
            if self.check_coup(grille, ligne+1,col-1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne+1)+str(col-1))
            if self.check_coup(grille, ligne-1,col+1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne-1)+str(col+1))

        if piece_couleur == Couleur.BLACK:
            if self.check_coup(grille, ligne-1,col-1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne-1)+str(col-1))
            if self.check_coup(grille, ligne+1,col+1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne+1)+str(col+1))
            if self.check_coup(grille, ligne+1,col-1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne+1)+str(col-1))
            if self.check_coup(grille, ligne-1,col+1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne-1)+str(col+1))

        return list_coup
    
    def _coupcheval(self, grille, piece_couleur, col, ligne):
        list_coup = []
        #Va tout droit puis en diagonale sauf si bloqué
        if self.check_coup(grille, ligne-1,col,piece_couleur, [0,self.lignes],[0,self.colonnes],elephant_block=True):
            if self.check_coup(grille, ligne-2,col-1,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col-1))
            if self.check_coup(grille, ligne-2,col+1,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col+1))    
        if self.check_coup(grille, ligne+1,col,piece_couleur, [0,self.lignes],[0,self.colonnes],elephant_block=True):
            if self.check_coup(grille, ligne+2,col-1,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col-1))
            if self.check_coup(grille, ligne+2,col+1,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col+1))
        if self.check_coup(grille, ligne,col-1,piece_couleur, [0,self.lignes],[0,self.colonnes],elephant_block=True):
            if self.check_coup(grille, ligne-1,col-2,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-1)+str(col-2))
            if self.check_coup(grille, ligne+1,col-2,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+1)+str(col-2))    
        if self.check_coup(grille, ligne,col+1,piece_couleur, [0,self.lignes],[0,self.colonnes],elephant_block=True):
            if self.check_coup(grille, ligne-1,col+2,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-1)+str(col+2))
            if self.check_coup(grille, ligne+1,col+2,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+1)+str(col+2))

        return list_coup
    
    def _couproi(self, grille, piece_couleur, col, ligne):
        list_coup = []
        if piece_couleur == Couleur.RED:
            #Va tout droit dans un périmètre restreint
            if self.check_coup(grille, ligne-1,col,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne-1)+str(col))
            if self.check_coup(grille, ligne+1,col,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne+1)+str(col))
            if self.check_coup(grille, ligne,col-1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne)+str(col-1))
            if self.check_coup(grille, ligne,col+1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne)+str(col+1))

            #Tu peux tuer le roi adverse avec ton roi en confrontation directe (l'un en face de l'autre sans pièce entre les deux)
            lK=ligne-1
            while lK > 0 and grille[lK][col]=="-":
                lK-=1
            if grille[lK][col]!="-":
                if grille[lK][col].type == PieceType.GENERAL:
                    list_coup.append(str(lK)+str(col))

        if piece_couleur == Couleur.BLACK:
            if self.check_coup(grille, ligne-1,col,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne-1)+str(col))
            if self.check_coup(grille, ligne+1,col,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne+1)+str(col))
            if self.check_coup(grille, ligne,col-1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne)+str(col-1))
            if self.check_coup(grille, ligne,col+1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne)+str(col+1))

            lK=ligne+1
            while lK < self.lignes - 1 and grille[lK][col]=="-":
                lK+=1
            if grille[lK][col]!="-":
                if grille[lK][col].type == PieceType.GENERAL:
                    list_coup.append(str(lK)+str(col))

        return list_coup
    
    def _couptour(self, grille, piece_couleur, col, ligne):
        #Va tout droit
        list_coup = []

        if ligne < self.lignes - 1:
            lK=ligne+1
            while lK < self.lignes - 1 and grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
                lK+=1
            if self.check_coup(grille, lK,col,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(lK)+str(col))

        if ligne > 0:
            lK=ligne-1
            while lK > 0 and grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
                lK-=1
            if self.check_coup(grille, lK,col,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(lK)+str(col))

        if col < self.colonnes - 1:
            cK=col+1
            while cK < self.colonnes - 1 and grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
                cK+=1
            if self.check_coup(grille, ligne,cK,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne)+str(cK))

        if col > 0:
            cK=col-1
            while cK > 0 and grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
                cK-=1
            if self.check_coup(grille, ligne,cK,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne)+str(cK))

        return list_coup
    
    def _coupcanon(self, grille, piece_couleur, col, ligne):
        #Va tout droit, peut sauter une pièce pour la manger
        list_coup = []

        if ligne < self.lignes-1:
            lK=ligne+1
            while lK < self.lignes - 1 and grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
                lK+=1
            if grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
            else:
                lK+=1
                while lK < self.lignes - 1 and grille[lK][col]=="-":
                    lK+=1
                if self.check_coup(grille, lK,col,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                    if grille[lK][col]!="-":
                        list_coup.append(str(lK)+str(col))

        if ligne > 0:
            lK=ligne-1
            while lK > 0 and grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
                lK-=1
            if grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
            else:
                lK-=1
                while lK > 0 and grille[lK][col]=="-":
                    lK-=1    
                if self.check_coup(grille, lK,col,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                    if grille[lK][col]!="-":
                        list_coup.append(str(lK)+str(col))

        if col < self.colonnes - 1:
            cK=col+1
            while cK < self.colonnes - 1 and grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
                cK+=1
            if grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
            else:
                cK+=1
                while cK < self.colonnes - 1 and grille[ligne][cK]=="-":
                    cK+=1
                if self.check_coup(grille, ligne,cK,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                    if grille[ligne][cK]!="-":
                        list_coup.append(str(ligne)+str(cK))

        if col > 0:
            cK=col-1
            while cK > 0 and grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
                cK-=1
            if grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
            else:
                cK-=1
                while cK > 0 and grille[ligne][cK]=="-":
                    cK-=1    
                if self.check_coup(grille, ligne,cK,piece_couleur,[0,self.lignes],[0,self.colonnes]):
                    if grille[ligne][cK]!="-":
                        list_coup.append(str(ligne)+str(cK))

        return list_coup


    def check_coup(self, grille, l, c, couleur, x_range, y_range, elephant_block=False):
        #Si la case est vide où qu'on peut la manger, on peut y aller
        if x_range[0] <= l < x_range[1] and y_range[0] <= c < y_range[1]:
            piece = grille[l][c]
            if piece == "-":
                return True
            
            elif couleur != piece.couleur and not(elephant_block):
                return True
        
        return False