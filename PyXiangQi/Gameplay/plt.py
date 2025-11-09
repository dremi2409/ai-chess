from enum import Enum
import numpy as np
from typing import Optional
from itertools import chain

class Couleur(Enum):
    RED = "RED"
    BLACK = "black"

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
    def __init__(self, type_piece: PieceType, couleur: Couleur, memory_access_number: int):
        self.type = type_piece
        self.couleur = couleur
        self.a_bouge = False  # Pour le roque et la première avancée du pion
        self.man = memory_access_number
    
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
    def __init__(self, lignes, colonnes):
        self.lignes = lignes
        self.colonnes = colonnes
        self.victoire = None

        self.grille = np.array([[Piece(PieceType.CHARIOT, Couleur.BLACK,31),Piece(PieceType.CHEVAL, Couleur.BLACK,30),
                                 Piece(PieceType.ELEPHANT, Couleur.BLACK,29),Piece(PieceType.ASSISTANT, Couleur.BLACK,28),
                                 Piece(PieceType.GENERAL, Couleur.BLACK,27),
                                 Piece(PieceType.ASSISTANT, Couleur.BLACK,26),Piece(PieceType.ELEPHANT, Couleur.BLACK,25),
                                 Piece(PieceType.CHEVAL, Couleur.BLACK,24),Piece(PieceType.CHARIOT, Couleur.BLACK,23)],
                                ["-","-","-","-","-","-","-","-","-"],
                                ["-",Piece(PieceType.CANON, Couleur.BLACK,22),"-","-","-","-","-",
                                 Piece(PieceType.CANON, Couleur.BLACK,21),"-"],
                                [Piece(PieceType.SOLDAT, Couleur.BLACK,20),"-",Piece(PieceType.SOLDAT, Couleur.BLACK, 19),"-",
                                 Piece(PieceType.SOLDAT, Couleur.BLACK,18),"-",Piece(PieceType.SOLDAT, Couleur.BLACK,17),"-",
                                 Piece(PieceType.SOLDAT, Couleur.BLACK,16)],
                                ["-","-","-","-","-","-","-","-","-"],
                                ["-","-","-","-","-","-","-","-","-"],
                                [Piece(PieceType.SOLDAT, Couleur.RED,15),"-",Piece(PieceType.SOLDAT, Couleur.RED,14),
                                 "-",Piece(PieceType.SOLDAT, Couleur.RED,13),"-",Piece(PieceType.SOLDAT, Couleur.RED,12),
                                 "-",Piece(PieceType.SOLDAT, Couleur.RED,11)],
                                ["-",Piece(PieceType.CANON, Couleur.RED,10),"-","-","-","-","-",
                                 Piece(PieceType.CANON, Couleur.RED,9),"-"],
                                ["-","-","-","-","-","-","-","-","-"],
                                [Piece(PieceType.CHARIOT, Couleur.RED,8),Piece(PieceType.CHEVAL, Couleur.RED,7),
                                 Piece(PieceType.ELEPHANT, Couleur.RED,6),Piece(PieceType.ASSISTANT, Couleur.RED,5),
                                 Piece(PieceType.GENERAL, Couleur.RED,4),
                                 Piece(PieceType.ASSISTANT, Couleur.RED,3),Piece(PieceType.ELEPHANT, Couleur.RED,2),
                                 Piece(PieceType.CHEVAL, Couleur.RED,1),Piece(PieceType.CHARIOT, Couleur.RED,0)]
                                ])
        
        self.tour_actuel = Couleur.RED
        self.grille_coups = [[self.calcul_coups_legaux(col,ligne) for col in range(self.colonnes)] for ligne in range(self.lignes)]
        
        self.historique = ["".join(list(map(lambda x: x if x=="-" else x.symbole(), chain.from_iterable(self.grille))))]

    def obtenir_piece_inverse(self, col: int, ligne: int) -> Optional[Piece]:
        """Retourne la pièce à une position donnée"""
        if 0 <= ligne < self.lignes and 0 <= col < self.colonnes:
            return self.grille[self.lignes - 1 - ligne][self.colonnes - 1 - col]
        return None
    
    def obtenir_piece(self, col: int, ligne: int) -> Optional[Piece]:
        """Retourne la pièce à une position donnée"""
        if 0 <= ligne < self.lignes and 0 <= col < self.colonnes:
            return self.grille[ligne][col]
        return None
    
    def deplacer_piece(self, init_col: int,init_ligne: int, col: int,ligne: int) -> None:
        self.grille[ligne][col] = self.grille[init_ligne][init_col] 
        self.grille[init_ligne][init_col] = "-"

        if self.tour_actuel == Couleur.RED:
            self.tour_actuel = Couleur.BLACK

        else:
            self.tour_actuel = Couleur.RED

        self.grille_coups = [[self.calcul_coups_legaux(col,ligne) for col in range(self.colonnes)] for ligne in range(self.lignes)]

        #Créer une liste "id" pour l'historique, les pats, et le ML
        nouvelle_position =  "".join(list(map(lambda x: x if x=="-" else x.symbole(), chain.from_iterable(self.grille))))
        if "將" not in nouvelle_position:
            self.victoire = Couleur.RED
        if "帥" not in nouvelle_position:
            self.victoire = Couleur.BLACK
        
        self.historique.append(nouvelle_position)
    
    def coup_legal(self, init_col: int,init_lin: int,col: int,ligne: int):
        return True
    
    def calcul_coups_legaux(self, col: int, ligne: int):
        piece = self.obtenir_piece(col,ligne)
        if piece != "-":
            piece_info = piece.type
            piece_couleur = piece.couleur

            if piece_couleur == self.tour_actuel:
                match piece_info:
                    case PieceType.SOLDAT:
                        list_coup = self._coupsoldat(piece_couleur, col, ligne)
                    case PieceType.ELEPHANT:
                        list_coup = self._coupelephant(piece_couleur, col, ligne)
                    case PieceType.ASSISTANT:
                        list_coup = self._coupassistant(piece_couleur, col, ligne) 
                    case PieceType.GENERAL:
                        list_coup = self._couproi(piece_couleur, col, ligne) 
                    case PieceType.CHEVAL:
                        list_coup = self._coupcheval(piece_couleur, col, ligne)
                    case PieceType.CHARIOT:
                        list_coup = self._couptour(piece_couleur, col, ligne)
                    case PieceType.CANON:
                        list_coup = self._coupcanon(piece_couleur, col, ligne)
                    
                    case _:
                        return None         
            else:
                return None    
        else:
            return None            
        if len(list_coup)==0:
            return None
        return list_coup
                    

    def _coupsoldat(self, piece_couleur, col, ligne):
        list_coup = []
        if piece_couleur == Couleur.RED:
            #Va tout droit si pas d'obstacle (piece de même couleur)
            if self.check_coup(ligne-1,col,Couleur.RED, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-1)+str(col))
            #Va sur les côtés après la rivière
            if self.check_coup(ligne,col+1,Couleur.RED, [0,5],[0,self.colonnes]):
                    list_coup.append(str(ligne)+str(col+1))
            if self.check_coup(ligne,col-1,Couleur.RED, [0,5],[0,self.colonnes]):
                        list_coup.append(str(ligne)+str(col-1))

        if piece_couleur == Couleur.BLACK:
            if self.check_coup(ligne+1,col,Couleur.BLACK, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+1)+str(col))
            if self.check_coup(ligne,col+1,Couleur.BLACK, [5,10],[0,self.colonnes]):
                    list_coup.append(str(ligne)+str(col+1))
            if self.check_coup(ligne,col-1,Couleur.BLACK, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne)+str(col-1))

        return list_coup
    
    def _coupelephant(self, piece_couleur, col, ligne):
        list_coup = []
        if piece_couleur == Couleur.RED:
            #Va en diagonale de deux cases sauf si bloqué
            if self.check_coup(ligne-1,col-1,Couleur.RED, [5,10],[0,self.colonnes],elephant_block=True) and self.check_coup(ligne-2,col-2,Couleur.RED, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col-2))
            if self.check_coup(ligne+1,col+1,Couleur.RED, [5,10],[0,self.colonnes],elephant_block=True) and self.check_coup(ligne+2,col+2,Couleur.RED, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col+2))
            if self.check_coup(ligne+1,col-1,Couleur.RED, [5,10],[0,self.colonnes],elephant_block=True) and self.check_coup(ligne+2,col-2,Couleur.RED, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col-2))
            if self.check_coup(ligne-1,col+1,Couleur.RED, [5,10],[0,self.colonnes],elephant_block=True) and self.check_coup(ligne-2,col+2,Couleur.RED, [5,10],[0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col+2))

        if piece_couleur == Couleur.BLACK:
            if self.check_coup(ligne-1,col-1,Couleur.BLACK, [0,5], [0,self.colonnes], elephant_block=True) and self.check_coup(ligne-2,col-2,Couleur.BLACK, [0,5], [0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col-2))
            if self.check_coup(ligne+1,col+1,Couleur.BLACK, [0,5], [0,self.colonnes], elephant_block=True) and self.check_coup(ligne+2,col+2,Couleur.BLACK, [0,5], [0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col+2))
            if self.check_coup(ligne+1,col-1,Couleur.BLACK, [0,5], [0,self.colonnes], elephant_block=True) and self.check_coup(ligne+2,col-2,Couleur.BLACK, [0,5], [0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col-2))
            if self.check_coup(ligne-1,col+1,Couleur.BLACK, [0,5], [0,self.colonnes], elephant_block=True) and self.check_coup(ligne-2,col+2,Couleur.BLACK, [0,5], [0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col+2))

        return list_coup
    
    def _coupassistant(self, piece_couleur, col, ligne):
        list_coup = []
        if piece_couleur == Couleur.RED:
            #Va en diagonale dans un périmètre restreint
            if self.check_coup(ligne-1,col-1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne-1)+str(col-1))
            if self.check_coup(ligne+1,col+1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne+1)+str(col+1))
            if self.check_coup(ligne+1,col-1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne+1)+str(col-1))
            if self.check_coup(ligne-1,col+1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne-1)+str(col+1))

        if piece_couleur == Couleur.BLACK:
            #Va en diagonale de deux cases sauf si bloqué
            if self.check_coup(ligne-1,col-1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne-1)+str(col-1))
            if self.check_coup(ligne+1,col+1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne+1)+str(col+1))
            if self.check_coup(ligne+1,col-1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne+1)+str(col-1))
            if self.check_coup(ligne-1,col+1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne-1)+str(col+1))

        return list_coup
    
    def _coupcheval(self, piece_couleur, col, ligne):
        list_coup = []
        #Va tout droit puis en diagonale sauf si bloqué
        if self.check_coup(ligne-1,col,piece_couleur, [0,self.lignes],[0,self.colonnes],elephant_block=True):
            if self.check_coup(ligne-2,col-1,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col-1))
            if self.check_coup(ligne-2,col+1,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-2)+str(col+1))    
        if self.check_coup(ligne+1,col,piece_couleur, [0,self.lignes],[0,self.colonnes],elephant_block=True):
            if self.check_coup(ligne+2,col-1,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col-1))
            if self.check_coup(ligne+2,col+1,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+2)+str(col+1))
        if self.check_coup(ligne,col-1,piece_couleur, [0,self.lignes],[0,self.colonnes],elephant_block=True):
            if self.check_coup(ligne-1,col-2,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-1)+str(col-2))
            if self.check_coup(ligne+1,col-2,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+1)+str(col-2))    
        if self.check_coup(ligne,col+1,piece_couleur, [0,self.lignes],[0,self.colonnes],elephant_block=True):
            if self.check_coup(ligne-1,col+2,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne-1)+str(col+2))
            if self.check_coup(ligne+1,col+2,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne+1)+str(col+2))

        return list_coup
    
    def _couproi(self, piece_couleur, col, ligne):
        list_coup = []
        if piece_couleur == Couleur.RED:
            #Va en diagonale dans un périmètre restreint
            if self.check_coup(ligne-1,col,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne-1)+str(col))
            if self.check_coup(ligne+1,col,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne+1)+str(col))
            if self.check_coup(ligne,col-1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne)+str(col-1))
            if self.check_coup(ligne,col+1,Couleur.RED,[7,10],[3,6]):
                list_coup.append(str(ligne)+str(col+1))

            #Tu peux tuer le roi adverse avec ton roi en confrontation directe (l'un en face de l'autre sans pièce entre les deux)
            lK=ligne-1
            while lK > 0 and self.grille[lK][col]=="-":
                lK-=1
            if self.grille[lK][col]!="-":
                if self.grille[lK][col].type == PieceType.GENERAL:
                    list_coup.append(str(lK)+str(col))

        if piece_couleur == Couleur.BLACK:
            #Va en diagonale de deux cases sauf si bloqué
            if self.check_coup(ligne-1,col,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne-1)+str(col))
            if self.check_coup(ligne+1,col,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne+1)+str(col))
            if self.check_coup(ligne,col-1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne)+str(col-1))
            if self.check_coup(ligne,col+1,Couleur.BLACK,[0,3],[3,6]):
                list_coup.append(str(ligne)+str(col+1))

            lK=ligne+1
            while lK < self.lignes - 1 and self.grille[lK][col]=="-":
                lK+=1
            if self.grille[lK][col]!="-":
                if self.grille[lK][col].type == PieceType.GENERAL:
                    list_coup.append(str(lK)+str(col))

        return list_coup
    
    def _couptour(self, piece_couleur, col, ligne):
        #Va tout droit
        list_coup = []

        if ligne < self.lignes - 1:
            lK=ligne+1
            while lK < self.lignes - 1 and self.grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
                lK+=1
            if self.check_coup(lK,col,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(lK)+str(col))

        if ligne > 0:
            lK=ligne-1
            while lK > 0 and self.grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
                lK-=1
            if self.check_coup(lK,col,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(lK)+str(col))

        if col < self.colonnes - 1:
            cK=col+1
            while cK < self.colonnes - 1 and self.grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
                cK+=1
            if self.check_coup(ligne,cK,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne)+str(cK))

        if col > 0:
            cK=col-1
            while cK > 0 and self.grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
                cK-=1
            if self.check_coup(ligne,cK,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                list_coup.append(str(ligne)+str(cK))

        return list_coup
    
    def _coupcanon(self, piece_couleur, col, ligne):
        #Va tout droit, peut sauter une pièce pour la manger
        list_coup = []

        if ligne < self.lignes-1:
            lK=ligne+1
            while lK < self.lignes - 1 and self.grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
                lK+=1
            if self.grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
            else:
                lK+=1
                while lK < self.lignes - 1 and self.grille[lK][col]=="-":
                    lK+=1
                if self.check_coup(lK,col,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                    if self.grille[lK][col]!="-":
                        list_coup.append(str(lK)+str(col))

        if ligne > 0:
            lK=ligne-1
            while lK > 0 and self.grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
                lK-=1
            if self.grille[lK][col]=="-":
                list_coup.append(str(lK)+str(col))
            else:
                lK-=1
                while lK > 0 and self.grille[lK][col]=="-":
                    lK-=1    
                if self.check_coup(lK,col,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                    if self.grille[lK][col]!="-":
                        list_coup.append(str(lK)+str(col))

        if col < self.colonnes - 1:
            cK=col+1
            while cK < self.colonnes - 1 and self.grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
                cK+=1
            if self.grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
            else:
                cK+=1
                while cK < self.colonnes - 1 and self.grille[ligne][cK]=="-":
                    cK+=1
                if self.check_coup(ligne,cK,piece_couleur, [0,self.lignes],[0,self.colonnes]):
                    if self.grille[ligne][cK]!="-":
                        list_coup.append(str(ligne)+str(cK))

        if col > 0:
            cK=col-1
            while cK > 0 and self.grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
                cK-=1
            if self.grille[ligne][cK]=="-":
                list_coup.append(str(ligne)+str(cK))
            else:
                cK-=1
                while cK > 0 and self.grille[ligne][cK]=="-":
                    cK-=1    
                if self.check_coup(ligne,cK,piece_couleur,[0,self.lignes],[0,self.colonnes]):
                    if self.grille[ligne][cK]!="-":
                        list_coup.append(str(ligne)+str(cK))

        return list_coup


    def check_coup(self, l, c, couleur, x_range, y_range, elephant_block=False):
        #Si la case est vide où qu'on peut la manger, on peut y aller
        if x_range[0] <= l < x_range[1] and y_range[0] <= c < y_range[1]:
            piece = self.grille[l][c]
            if piece == "-":
                return True
            
            elif couleur != piece.couleur and not(elephant_block):
                return True
        
        return False