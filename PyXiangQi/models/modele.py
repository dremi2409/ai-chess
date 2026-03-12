from PyXiangQi.Gameplay.plt import Plateau, Couleur, PieceType
import random as rd
import time
import numpy as np
import torch
from DeepNorm import DeepNormClassifier, Eval_
from torchsummary import summary
import math
import os
import json
from copy import deepcopy

class Modele:
    #Interface graphique du XiangQi
    def __init__(self, model_name = None, modele_path=None, training=False, n_sim=10):
        self.name = model_name
        self.training = training
        self.nsim=n_sim

        if self.name=="Deepnorm":
            file_path = os.getcwd() + "\\PyXiangQi\\models\\Mapping.json"
            with open(file_path, 'r', encoding='utf-8') as file:
                self.map_AI = json.load(file)  # Parse JSON into Python object
                self.inverted_map = {v: k for k, v in self.map_AI.items()}

            #Charger le modèle IA
            # Config legere
            cfg = dict(
                vocab_size  = 100,
                max_seq_len = 1620,
                d_model     = 128,
                n_heads     = 4,
                n_layers    = 6,
                d_ff        = 512,
                dropout     = 0.1,
            )

            self.name = "IA"
            self.model = DeepNormClassifier(num_classes=4501, **cfg)
            self.eval=Eval_()
            self.model.load_state_dict(torch.load(modele_path, weights_only=True))

    #Demande au modèle le prochain coup parmis ceux possibles
    def trouver_coup(self, plateau: Plateau):
        match self.name:
            #Joueur aléatoire
            case "Aleatoire":
                #time.sleep(0.4)
                coup = rd.choice(plateau.get_coup_possible) 
                return int(coup[0]), int(coup[1]), int(coup[2]), int(coup[3])

            case "IA":
                if not(self.training):
                    input_tenseur = torch.from_numpy(self.plt_to_tensor(plateau))
                    self.model.eval()
                    with torch.no_grad():
                        eval_val, policy_val = self.model(input_tenseur)
                    
                    mask=torch.full((1, 4500), float('-inf'))
                    coup_autorises=self.map_coups(plateau.get_coup_possible)
                    mask[0][coup_autorises] = 0.0

                    preds=self.eval(policy_val=policy_val,mask=mask)
                    idxmax = torch.argmax(preds).item() 
                    
                    ligne = (idxmax // 50) % 10
                    colonne = idxmax // 500
                    print(self.inverted_map)
                    dx,dy = self.find_delta(self.inverted_map[idxmax%50])
                    li = ligne + dx
                    col = colonne + dy

                    print("Le coups choisi est : " + str(ligne) + str(colonne) + str(li) + str(col))
                    print(self.map_AI[str(dx)+str(dy)], idxmax)
                    return ligne, colonne, li, col
                
                else: #training
                    self.model.train()
                    childs={}     #stcke les chemins du graphe
                    dict_coups={plateau.historique[-1]:{"Q":0,"N":0}} #stocke les N-values, Q-values, etc. 
                    distrib_chemin=torch.full((1, 4500), float(0))  #stocke l'ensemble des premiers coups choisis

                    for a in range(self.nsim):
                        chemin=[]
                        self.MCGS(plateau, dict_coups, childs, chemin, root_node=True)
                        distrib_chemin[0][chemin[0]]=distrib_chemin[0][chemin[0]]+1/self.nsim

                    input_tenseur = torch.from_numpy(self.plt_to_tensor(plateau))
                    V, Pi = self.model(input_tenseur)
                    
                    mask=torch.full((1, 4500), float('-inf'))
                    coup_autorises=self.map_coups(plateau.get_coup_possible)
                    mask[0][coup_autorises] = 0.0

                    P=self.eval(policy_val=Pi,mask=mask)
                    
                    return ligne, colonne, li, col, P, distrib_chemin, V
          
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

    def MCGS(self, plt, coups, children, way, root_node=False):
        init_hash = plt.historique[-1]
        way.append(init_hash)
        N_node=coups[init_hash]["N"]

        if (N_node==0 and not(root_node)) or plt.victoire==Couleur.RED or plt.victoire==Couleur.GREY or plt.victoire==Couleur.BLACK:
            V=coups[init_hash]["Q"]
            for idx in range(len(way)):
                hash=way[-(idx+1)]
                if idx>0:
                    coups[hash]["Q"]=(coups[hash]["Q"]*coups[hash]["N"]+V*(-1)**idx)/(coups[hash]["N"]+1) 
                coups[hash]["N"]=coups[hash]["N"]+1
            print(coups, children, way)


        else:
            input_tenseur = torch.from_numpy(self.plt_to_tensor(plt))
            V, Pi = self.model(input_tenseur)

            mask=torch.full((1, 4500), float('-inf'))
            coup_autorises=self.map_coups(plt.get_coup_possible)
            mask[0][coup_autorises] = 0.0   
            
            P=self.eval(policy_val=Pi,mask=mask)
            mask_P=torch.full((1, 4500), float('-inf'))
            mask_Q=torch.full((1, 4500), float(0))
            
            if root_node:
                noise = torch.distributions.Dirichlet(torch.tensor([0.3 for _ in range(len(coup_autorises))])).sample()
                mask_P[0][coup_autorises] = noise 
                # Mix original tensor with noise
                P = torch.add((1 - 0.25) * P, 0.25 * mask_P)
            
            N_tot = 0
            try:
                for child_info in children[init_hash]:
                    pos=child_info["pos"]
                    hash=child_info["hash"]

                    Q=coups[hash]["Q"]
                    N=coups[hash]["N"]
                    N_tot+=N

                    mask_Q[0][pos] = Q
                    P[0][pos] = P[0][pos]/(1+N)

            except:
                list_childs=[]
                for cp in coup_autorises:
                    ligne = (cp // 50) % 10
                    colonne = cp // 500
                    dx,dy = self.find_delta(self.inverted_map[cp%50])
                    li = ligne + dx
                    col = colonne + dy

                    plt_copy = Plateau(lignes=10, colonnes=9, plt=deepcopy(plt.grille), 
                                        historique=deepcopy(plt.historique), tour_actuel=deepcopy(plt.tour_actuel)) 
                    plt_copy.deplacer_piece(colonne, ligne, col, li)
                    hash = plt_copy.historique[-1]
                    list_childs.append({"pos":cp, "hash":hash})

                    try:
                        Q=coups[hash]["Q"]
                        N=coups[hash]["N"]

                    except:
                        if plt.victoire==Couleur.RED or plt.victoire==Couleur.BLACK:
                            Val=-1
                        elif plt.victoire==Couleur.GREY:
                            Val=0
                        else:
                            input_tenseur = torch.from_numpy(self.plt_to_tensor(plt))
                            Val, Pi = self.model(input_tenseur)
                        Q=Val
                        N=0
                        coups[hash]={}
                        coups[hash]["Q"]=Q
                        coups[hash]["N"]=N
                        
                    N_tot+=N
                    mask_Q[0][cp] = Q
                    P[0][cp]=P[0][cp]/(N+1)

                children[init_hash]=list_childs
                if N_tot>0:
                    PUCT=torch.add(-mask_Q,torch.mul(P,math.sqrt(N_tot)))
                else:
                    PUCT=-mask_Q

                max_PUCT=torch.argmax(PUCT).item()
                
                ligne = (max_PUCT // 50) % 10
                colonne = max_PUCT // 500
                dx,dy = self.find_delta(self.inverted_map[max_PUCT%50])
                li = ligne + dx
                col = colonne + dy

                plt_copy = Plateau(lignes=10, colonnes=9, plt=deepcopy(plt.grille), 
                                historique=deepcopy(plt.historique), tour_actuel=deepcopy(plt.tour_actuel)) 
                
                
                print(torch.argmax(PUCT))
                print(plt_copy.tour_actuel,plt_copy.historique[-1])
                print(colonne, ligne, col, li)
                plt_copy.deplacer_piece(colonne, ligne, col, li)
                print(plt_copy.tour_actuel,plt_copy.historique[-1])

            self.MCGS(plt_copy,coups,children,way)

                    
    def map_coups(self, grille):
        mapping=[]

        for coup in grille:
            ligne, colonne, li, col = int(coup[0]), int(coup[1]), int(coup[2]), int(coup[3])
            str_coup=str(li-ligne)+str(col-colonne)
            mapping.append(self.map_AI[str_coup]+ligne*50+colonne*500)

        return mapping

    def plt_to_tensor(self, plt: Plateau):
        #Transforme la position en grille 10*9*14 (Lignes - colonnes - pièces différentes)
        grille =plt.grille
        hist=plt.historique

        numpy_array_repet = np.zeros((plt.lignes*plt.colonnes,14+4))

        #Ajouter les pièces au tenseur
        for pos in range(len(hist[-1:])):
            plt_ref = hist[-pos-1]
            for ligne in range(plt.lignes):
                for col in range(plt.colonnes):
                    piece = plt_ref[ligne*9+col]
                    if piece != '-':
                        match piece:
                            case "將":
                                base = 0
                            case "仕":
                                base = 1
                            case "象":
                                base = 2
                            case "傌":
                                base = 3
                            case "俥":
                                base = 4
                            case "砲":
                                base = 5
                            case "卒":
                                base = 6
                            case "帥":
                                base = 7
                            case "士":
                                base = 8
                            case "相":
                                base = 9
                            case "馬":
                                base = 10
                            case "車":
                                base = 11
                            case "炮":
                                base = 12
                            case "兵":
                                base = 13

                        numpy_array_repet[ligne*9+col][base] = 1

                    #Ajouter le tour du joueur
                    match plt.tour_actuel:
                        case Couleur.RED:
                            coeff = 14

                        case Couleur.BLACK:
                            coeff = 15

                    numpy_array_repet[ligne*9+col][coeff] = 1 

                    #Regarder les répetitions et le nombre de coups sans manger une pièce 
                    numpy_array_repet[ligne*9+col][16]= plt.pas_de_prise
                    numpy_array_repet[ligne*9+col][17] = plt.nombre_repet

        return np.array([numpy_array_repet.flatten()],dtype=np.int32)

    def find_delta(self, delta):
        n_char=len(delta)
        if n_char==2:
            return int(delta[0]), int(delta[1])
        elif n_char==4:
            return -int(delta[1]), -int(delta[3])
        elif delta[0]=="-":
            return -int(delta[1]), int(delta[2])
        else:
            return int(delta[0]), -int(delta[2])