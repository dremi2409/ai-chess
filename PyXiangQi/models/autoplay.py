import os
from itertools import chain
import numpy as np
import sys
import time
import torch
sys.path.append('C://Users//guigz//OneDrive//Documents//Projets//ai-chess')

from PyXiangQi.Gameplay.plt import Plateau, Couleur
from PyXiangQi.models.modele import Modele
from DeepNorm import CombinedLoss

class Autoplay_xiangqi:
    #Interface graphique du XiangQi
    def __init__(self, lignes: int = 10, colonnes: int = 9,
                    Plt=None, Joueur1='humain', Joueur2='Aleatoire', 
                        modele_path=None, modele_path2=None, training=False,
                            n_games=1, save_path=None):

        #Joueur 1 : Rouge / Joueur 2 : Noir
        self.J1 = Joueur1
        self.J2 = Joueur2

        self.modele_rouge = Modele(model_name=Joueur1, 
                                   modele_path=modele_path, training=training)
        self.modele_noir = Modele(model_name=Joueur2, 
                                  modele_path=modele_path2, training=training)

        self.lignes = lignes
        self.colonnes = colonnes

        for _ in range(n_games):
            if training:
                P_list=[]
                Pi_list=[]
                V_list=[]
                optimizer = torch.optim.Adam(self.modele_rouge.model.parameters(), weight_decay=10**(-4))           

            if Plt:
                self.plateau = Plt
            else:
                self.plateau = Plateau(self.lignes, self.colonnes)
            self.dessiner_plateau()

            cont_coup = 0
            while not(self.plateau.victoire): 
                if not(training):
                    if cont_coup%100==0:
                        print(cont_coup)
                    cont_coup+=1

                    if self.plateau.tour_actuel == Couleur.RED and self.J1 == "humain" or self.plateau.tour_actuel == Couleur.BLACK and self.J2 == "humain":
                        self.dessiner_plateau()

                        deplacement = None
                        while not(deplacement):
                            if self.plateau.tour_actuel == Couleur.RED:
                                coup=input("A vous de jouer ! Rouge :")
                            else:
                                coup=input("A vous de jouer ! Noir :")

                            deplacement=self.parse(coup)
                            print(deplacement)
                            
                        print(self.unparse(deplacement))
                    
                        self.plateau.deplacer_piece(deplacement[0],deplacement[1],deplacement[2],deplacement[3])
                        self.dernier_coup = deplacement
                        self.dessiner_plateau()
                        
                        if self.plateau.victoire == Couleur.RED:
                            # Message de victoire
                            print("\n\nVictoire rouge !")

                        elif self.plateau.victoire == Couleur.BLACK:
                            # Message de victoire
                            print("\n\nVictoire noire !")

                        elif self.plateau.victoire == Couleur.GREY:
                            # Message de victoire
                            print("\n\nNulle...")

                    else:
                        if self.plateau.tour_actuel == Couleur.RED:
                                y_init, x_init, y, x = self.modele_rouge.trouver_coup(self.plateau)
                        else:
                            y_init, x_init, y, x = self.modele_noir.trouver_coup(self.plateau)
                        self.plateau.deplacer_piece(x_init,y_init,x,y)

                        if self.plateau.victoire == Couleur.RED:
                            self.dessiner_plateau()
                            # Message de victoire
                            print("\n\nVictoire rouge !")

                        elif self.plateau.victoire == Couleur.BLACK:
                            self.dessiner_plateau()
                            # Message de victoire
                            print("\n\nVictoire noire !") 

                        elif self.plateau.victoire == Couleur.GREY:
                            self.dessiner_plateau()
                            # Message de victoire
                            print("\n\nNulle...")

                else: #training
                    y_init, x_init, y, x, P, PI, V = self.modele_rouge.trouver_coup(self.plateau)
                    P_list.append(P)
                    Pi_list.append(PI)
                    V_list.append(V)
                    
                    self.plateau.deplacer_piece(x_init,y_init,x,y)

                    if self.plateau.victoire == Couleur.RED:
                        self.dessiner_plateau()
                        # Message de victoire
                        print("\n\nVictoire rouge !")

                        Vi_list=np.array([(-1.0)**idx for idx in range(len(V))])
                        V_list=np.array(V_list)
                        print(P_list,Pi_list,Vi_list,V_list)
                        loss=CombinedLoss()
                        output=loss(torch.stack(Pi_list),torch.stack(P_list),torch.from_numpy(Vi_list),torch.from_numpy(V_list))
                        output.backward()                 # calcule les gradients
                        optimizer.step()                  # met à jour les poids

                    elif self.plateau.victoire == Couleur.BLACK:
                        self.dessiner_plateau()
                        # Message de victoire
                        print("\n\nVictoire noire !")

                        Vi_list=np.array([(-1.0)**(idx+1) for idx in range(len(V))])
                        V_list=np.array(V_list)
                        print(P_list,Pi_list,Vi_list,V_list)
                        loss=CombinedLoss()
                        output=loss(torch.stack(Pi_list),torch.stack(P_list),torch.from_numpy(Vi_list),torch.from_numpy(V_list))
                        output.backward()                 # calcule les gradients
                        optimizer.step()                  # met à jour les poids

                    elif self.plateau.victoire == Couleur.GREY:
                        self.dessiner_plateau()
                        # Message de victoire
                        print("\n\nNulle...")

                        Vi_list=np.array([0.0 for _ in range(len(V))])
                        V_list=np.array(V_list)
                        print(P_list,Pi_list,Vi_list,V_list)
                        loss=CombinedLoss()
                        output=loss(torch.stack(Pi_list),torch.stack(P_list),torch.from_numpy(Vi_list),torch.from_numpy(V_list))
                        output.backward()                 # calcule les gradients
                        optimizer.step()                  # met à jour les poids

    def parse(self, input):
        try:
            plateau = self.plateau.grille

            if self.plateau.tour_actuel == Couleur.RED:
                plt = np.array([[plateau[len(plateau)-idx-1][len(plateau[0])-jdx-1] for jdx in range(len(plateau[0]))] for idx in range(len(plateau))])
            
            if input[0] == "+" or input[0]=="-" or input[1]=="P":
                try:
                    pos=int(input[0])
                except:
                    pos=input[0]
                valeur=input[1]      
                input=input[2:]
                
            else:
                pos=False
                valeur=input[0]
                input=input[1:]

            colonne=int(input[0])-1
            ligne=False
            movement=input[1]
            amplitude=int(input[2])

            #Locate piece
            compt=1
            idx=len(plt)-1
            for piece in np.flip(plt[:,colonne]):
                if type(piece)!=str:
                    if piece.couleur == self.plateau.tour_actuel and piece.symbole_eng() == valeur:
                        if not(pos) or pos==str(compt) or (pos=="+" and compt==1) or (pos=="-" and compt==2):
                            ligne=idx       
                        compt+=1
                idx-=1

            if not(type(ligne)==int):
                return False
            
            #Cas des coups en ligne
            if valeur in ["K","R","C","P"]:
                if movement=="+":
                    li=ligne+amplitude
                    col=colonne
                elif movement=="-":
                    li=ligne-amplitude
                    col=colonne
                else:
                    li=ligne
                    col=amplitude-1

            #Cas du cavalier
            if valeur=="H":
                if movement=="+":
                    col=amplitude-1
                    if abs(colonne-col)==1:
                        li=ligne+2
                    else:
                        li=ligne+1
                elif movement=="-":
                    col=amplitude-1
                    if abs(colonne-col)==1:
                        li=ligne-2
                    else:
                        li=ligne-1

            #Cas du conseiller
            if valeur=="A":
                if movement=="+":
                    li=ligne+1
                    col=amplitude-1
                elif movement=="-":
                    li=ligne-1
                    col=amplitude-1

            #Cas de l'elephant
            if valeur=="E":
                if movement=="+":
                    li=ligne+2
                    col=amplitude-1
                elif movement=="-":
                    li=ligne-2
                    col=amplitude-1

            #Inverser les pos pour le joueur rouge
            if self.plateau.tour_actuel == Couleur.RED:
                ligne, colonne, li, col = self.lignes-ligne-1, self.colonnes-colonne-1, self.lignes-li-1, self.colonnes-col-1

            #Vérifier si le coup est légal
            if str(ligne)+str(colonne)+str(li)+str(col) not in self.plateau.get_coup_possible:
                print(self.plateau.get_coup_possible)
                print(ligne, colonne, li, col)
                return False

            return [colonne, ligne, col, li]
        
        except:
            return False
            
    def unparse(self, coup):
        notation = ""
        plateau = self.plateau.grille

        if self.plateau.tour_actuel == Couleur.RED:
            plt = np.array([[plateau[len(plateau)-idx-1][len(plateau[0])-jdx-1] for jdx in range(len(plateau[0]))] for idx in range(len(plateau))])
            ligne,colonne,li,col = self.plateau.lignes-coup[1]-1, self.plateau.colonnes-coup[0]-1, self.plateau.lignes-coup[3]-1, self.plateau.colonnes-coup[2]-1

        else:
            plt = np.array([[plateau[idx][jdx] for jdx in range(len(plateau[0]))] for idx in range(len(plateau))])
            ligne,colonne,li,col = coup[1], coup[0], coup[3], coup[2]

        piece = plt[ligne][colonne]
        piece_nam = piece.symbole_eng()

        colonne_complete = plt[:,colonne]
        compt_piece=0
        for symb in colonne_complete:
            if type(symb)!=str:
                if symb.symbole() == piece.symbole():
                    compt_piece+=1

        if compt_piece>1:
            if piece_nam=="P":
                piecenum=1
                for idx in range(ligne+1,self.lignes):
                    if type(plt[idx][colonne])!=str:
                        if plt[idx][colonne].symbole()==piece.symbole():
                            piecenum+=1

                notation+=str(piecenum)
            
            else:
                firstp=True
                for idx in range(ligne+1,self.lignes):
                    if type(plt[idx][colonne])!=str:
                        if plt[idx][colonne].symbole()==piece.symbole():
                            firstp=False

                if firstp:
                    notation+="+"
                else:
                    notation+="-"

        if piece_nam in ["K","R","C","P"]:
            if colonne!=col:
                notation+=piece_nam+str(colonne+1)+"="+str(col+1)
            else:
                if ligne>li:
                    notation+=piece_nam+str(colonne+1)+"-"+str(ligne-li)
                else:
                    notation+=piece_nam+str(colonne+1)+"+"+str(li-ligne)

        else:
            if ligne>li:
                notation+=piece_nam+str(colonne+1)+"-"+str(col+1)
            else:
                notation+=piece_nam+str(colonne+1)+"+"+str(col+1)
            
        return notation

    def dessiner_plateau(self):
        """Dessine le plateau et les pièces"""      
        # Dessiner les pièces
        for ligne in range(self.lignes):
            row_char=""
            for col in range(self.colonnes):
                if self.plateau.grille[ligne][col] != "-":
                    row_char += self.plateau.grille[ligne][col].symbole() + " "

                else:
                    row_char += ("・ ")
                    
            print(row_char)
                
        
# Lancement de l'application
if __name__ == "__main__":
    t0=time.time()
    app = Autoplay_xiangqi(Joueur1='Deepnorm', Joueur2='humain', modele_path='PyXiangQi//models//weights//v0.pth', training=True)
    print(time.time()-t0)