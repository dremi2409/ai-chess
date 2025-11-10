import sys
from pathlib import Path
import time

# Ajoute le dossier parent 'mon_projet' au PYTHONPATH
projet_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(projet_root))

import tkinter as tk
from PIL import Image, ImageTk
import os
from itertools import chain
from tkinter import messagebox
from PyXiangQi.Gameplay.plt import Plateau, Couleur
from PyXiangQi.models.modele import Modele

image_path = "PyXiangQi\\GUI\\pieces_canvas"


class Interface_xiangqi:
    #Interface graphique du XiangQi
    def __init__(self, root: tk.Tk, lignes: int = 10, colonnes: int = 9, title = "让游戏开始!", 
                 taille_case = 70, Plt=None, Joueur1='humain', Joueur2='humain'):
        self.root = root
        self.root.title(title)
        self.taille_case = taille_case

        #Joueur 1 : Rouge / Joueur 2 : Noir
        self.J1 = Joueur1
        self.J2 = Joueur2

        self.modele_rouge = Modele(model_name=Joueur1)
        self.modele_noir = Modele(model_name=Joueur2)

        self.lignes = lignes
        self.colonnes = colonnes

        # Variables to track dragging
        self.drag_data = {"x": 0, "y": 0, "item": None, "col":0, "lin":0}
        self.img_ref = []

        if Plt:
            self.plateau = Plt
        else:
            self.plateau = Plateau(self.lignes, self.colonnes)

        if self.J1 == 'humain':
            self.position = "DROIT"
        else:
            self.position = "INVERSE"


        self.canvas = tk.Canvas(root, width=colonnes * self.taille_case, 
                                height=lignes * self.taille_case)
        self.canvas.pack()

        # Variables pour le drag & drop
        self.piece_selectionnee = None
        self.dernier_coup = None

        # Événement
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)

        if self.J1 == 'humain':
            self.canvas.bind("<ButtonRelease-1>", self.on_drag_stop)
            self.dessiner_plateau()

        else:
            self.canvas.bind("<ButtonRelease-1>", self.on_drag_stop_noclick)
            self.dessiner_plateau()
            self.coup_IA(modele=self.modele_rouge)
        
        
    def dessiner_plateau(self):
        """Dessine le plateau et les pièces"""
        self.canvas.delete("all")
        
        # Dessiner les cases
        for ligne in range(self.lignes):
            if self.position == "INVERSE":
                    ligne = self.lignes - ligne - 1

            for col in range(self.colonnes):
                if self.position == "INVERSE":
                    col = self.colonnes - col - 1

                x1 = col * self.taille_case
                y1 = ligne * self.taille_case
                x2 = x1 + self.taille_case
                y2 = y1 + self.taille_case
                
                couleur = "#B58863"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=couleur, outline="")

        #Laisser des traces sur le dernier coup joué
        if self.dernier_coup:
            ligne, col = int(self.dernier_coup[0]), int(self.dernier_coup[1])
            if self.position == "INVERSE":
                ligne = self.lignes - ligne - 1
                col = self.colonnes - col - 1

            couleur = "#C5B4A5"

            x1 = col * self.taille_case
            y1 = ligne * self.taille_case
            x2 = x1 + self.taille_case
            y2 = y1 + self.taille_case

            self.canvas.create_rectangle(x1, y1, x2, y2, fill=couleur, outline="")

            ligne, col = int(self.dernier_coup[2]), int(self.dernier_coup[3])
            if self.position == "INVERSE":
                ligne = self.lignes - ligne - 1
                col = self.colonnes - col - 1

            couleur = "#C5844C"

            x1 = col * self.taille_case
            y1 = ligne * self.taille_case
            x2 = x1 + self.taille_case
            y2 = y1 + self.taille_case

            self.canvas.create_rectangle(x1, y1, x2, y2, fill=couleur, outline="")


        #Dessiner les échecs
        if self.plateau.echec:
            ligne, col = int(self.plateau.echec[0]), int(self.plateau.echec[1])
            if self.position == "INVERSE":
                ligne = self.lignes - ligne - 1
                col = self.colonnes - col - 1

            couleur = "#f00020"

            x1 = col * self.taille_case
            y1 = ligne * self.taille_case
            x2 = x1 + self.taille_case
            y2 = y1 + self.taille_case

            self.canvas.create_rectangle(x1, y1, x2, y2, fill=couleur, outline="")

            for check_piece in self.plateau.piece_echec:
                ligne, col = int(check_piece[0]), int(check_piece[1])
                if self.position == "INVERSE":
                    ligne = self.lignes - ligne - 1
                    col = self.colonnes - col - 1

                couleur = "#1d4023"

                x1 = col * self.taille_case
                y1 = ligne * self.taille_case
                x2 = x1 + self.taille_case
                y2 = y1 + self.taille_case

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=couleur, outline="")

        for ligne in range(self.lignes):
            #Créer une ligne noire
            y = self.taille_case * ligne + self.taille_case // 2
            x1 = self.taille_case // 2
            x2 = self.taille_case * (self.colonnes - 1) + self.taille_case // 2
            self.canvas.create_line(x1, y, x2, y, width=2)

        for col in range(self.colonnes):
            #Créer deux colonnes séparées par la rivière
            x = self.taille_case * col + self.taille_case // 2
            y1 = self.taille_case // 2
            y2 = self.taille_case * (self.lignes // 2 - 1) + self.taille_case // 2

            y3 = self.taille_case * (self.lignes // 2) + self.taille_case // 2
            y4 = self.taille_case * (self.lignes - 1) + self.taille_case // 2
            
            self.canvas.create_line(x, y1, x, y2, width=2) 
            self.canvas.create_line(x, y3, x, y4, width=2) 

        #Créer les deux diagonales de l'espace royal
        x1 = (self.lignes // 2 + 0.5) * self.taille_case
        x2 = (self.lignes // 2 - 1.5) * self.taille_case 
        y1 = self.taille_case // 2
        y2 = 2 * self.taille_case + self.taille_case // 2
        self.canvas.create_line(x1, y1, x2, y2, width=3) 
        self.canvas.create_line(x2, y1, x1, y2, width=3)

        y3 = self.taille_case * (self.colonnes - 1) - self.taille_case // 2
        y4 = self.taille_case * (self.colonnes + 1) - self.taille_case // 2
        self.canvas.create_line(x1, y3, x2, y4, width=3) 
        self.canvas.create_line(x2, y3, x1, y4, width=3)

        #Caractère chinois rivière (楚河漢界)
        x = self.taille_case + self.taille_case // 2
        y = (self.lignes // 2) * self.taille_case
        self.canvas.create_text(x, y, text="楚", 
                                           font=("Arial", round(self.taille_case*0.6)), )
        
        x = self.taille_case * 3
        y = (self.lignes // 2) * self.taille_case
        self.canvas.create_text(x, y, text="河", 
                                           font=("Arial", round(self.taille_case*0.6)), )
        
        x = self.taille_case * 6
        y = (self.lignes // 2) * self.taille_case
        self.canvas.create_text(x, y, text="漢", 
                                           font=("Arial", round(self.taille_case*0.6)), )
        
        x = self.taille_case * 7 + self.taille_case // 2
        y = (self.lignes // 2) * self.taille_case
        self.canvas.create_text(x, y, text="界", 
                                           font=("Arial", round(self.taille_case*0.6)), )
                
        # Dessiner les pièces
        for ligne in range(self.lignes):
            ligne_2=ligne
            if self.position == "INVERSE":
                    ligne = self.lignes - ligne - 1

            for col in range(self.colonnes):
                col_2 = col
                if self.position == "INVERSE":
                    col = self.colonnes - col - 1
                
                piece = self.plateau.obtenir_piece(self.plateau.grille, col, ligne)
                if piece != "-":
                    image = ImageTk.PhotoImage(Image.open(os.path.join(image_path,piece.symbole()+".png")).resize((60,60)))
                    x = (col_2 * self.taille_case + self.taille_case // 2)
                    y = (ligne_2 * self.taille_case + self.taille_case // 2)
                        
                    # Add the image to the Canvas
                    self.canvas.create_image(x, y, image=image)  # Position at (200, 200)

                    # Keep a reference to the image to prevent garbage collection
                    self.img_ref.append(image)

    def dessiner_coups_possibles(self, col, ligne):
        for coup_possible in self.plateau.grille_coups[ligne][col]:
            image = ImageTk.PhotoImage(Image.open(os.path.join(image_path,"spot.png")).resize((20,20)))
            if self.position == "INVERSE":
                l = self.lignes - int(coup_possible[0]) - 1
                c = self.colonnes - int(coup_possible[1]) - 1

            else:
                l = int(coup_possible[0])
                c = int(coup_possible[1])

            x = (c * self.taille_case + self.taille_case // 2)
            y = (l * self.taille_case + self.taille_case // 2)
                        
            # Add the image to the Canvas
            self.canvas.create_image(x, y, image=image)  # Position at (200, 200)

            # Keep a reference to the image to prevent garbage collection
            self.img_ref.append(image)


    # Mouse events for dragging
    def on_drag_start(self, event):
        """Gère le clic initial pour sélectionner une pièce"""
        col = event.x // self.taille_case
        ligne = event.y // self.taille_case

        if self.position == "INVERSE":
                col = self.colonnes - col - 1
                ligne = self.lignes - ligne - 1

        #Enregistrer la position de base pour connaître les coups possibles
        self.drag_data["col"] = col
        self.drag_data["lin"] = ligne
        
        piece = self.plateau.obtenir_piece(self.plateau.grille, col, ligne)

        self.drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        item_type = self.canvas.type(self.drag_data["item"])

        if piece !="-" and piece and item_type == 'image':
            self.piece_selectionnee = piece
            if self.plateau.grille_coups[ligne][col] and ((self.plateau.tour_actuel == Couleur.RED and self.J1 == "humain") or (self.plateau.tour_actuel == Couleur.BLACK and self.J2 == "humain")):
                self.dessiner_coups_possibles(col, ligne)

            #mettre toujours la pièce bougée au dessus 
            self.canvas.tag_raise(self.canvas.find_closest(event.x, event.y)[0])

        else:
            self.drag_data["item"] = None


    def on_drag_motion(self, event):
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.canvas.move(self.drag_data["item"], dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_drag_stop(self, event):
        col = event.x // self.taille_case
        ligne = event.y // self.taille_case

        if self.position == "INVERSE":
                col = self.colonnes - col - 1
                ligne = self.lignes - ligne - 1

        if self.plateau.grille_coups[self.drag_data["lin"]][self.drag_data["col"]] and self.drag_data["item"] != None and 0 <= col < self.colonnes and 0 <= ligne < self.lignes :
            if str(ligne)+str(col) in self.plateau.grille_coups[self.drag_data["lin"]][self.drag_data["col"]]:
                self.plateau.deplacer_piece(self.drag_data["col"],self.drag_data["lin"],col,ligne)
                self.dernier_coup = [self.drag_data["lin"],self.drag_data["col"],ligne,col]
                self.dessiner_plateau()

                if self.J1 != self.J2:
                    self.canvas.update()
                
                if self.plateau.victoire == Couleur.RED:
                    # Message de victoire
                    messagebox.showinfo("Information", "Victoire rouge !")
                    # Fermer la fenêtre principale
                    root.destroy()

                elif self.plateau.victoire == Couleur.BLACK:
                    # Message de victoire
                    messagebox.showinfo("Information", "Victoire noire !")
                    # Fermer la fenêtre principale
                    root.destroy()

                elif self.plateau.victoire == Couleur.GREY:
                    # Message de victoire
                    messagebox.showinfo("Information", "Nulle...")
                    # Fermer la fenêtre principale
                    root.destroy()

                #Demander le coup à l'IA si l'adversaire n'est pas humain, enlever la fonction de choix
                elif self.J1 != self.J2:
                    self.canvas.update()
                    self.canvas.unbind("<ButtonRelease-1>")
                    self.canvas.bind("<ButtonRelease-1>", self.on_drag_stop_noclick)

                    if self.plateau.tour_actuel == Couleur.RED:
                        self.coup_IA(modele = self.modele_rouge)

                    else:
                        self.coup_IA(modele = self.modele_noir)
        
        #Arreter de déplacer la pièce
        self.drag_data["item"] = None

        #Retourner le plateau pour le joueur d'après
        if self.plateau.tour_actuel == Couleur.RED and self.J1 == "humain":
            self.position = "DROIT"
        if self.plateau.tour_actuel == Couleur.BLACK and self.J2 == "humain":
            self.position = "INVERSE"  

        self.dessiner_plateau()

    def coup_IA(self, modele: Modele):        
        y_init, x_init, y, x = modele.trouver_coup(self.plateau)
        self.plateau.deplacer_piece(x_init,y_init,x,y)
        self.dernier_coup = [y_init,x_init,y,x]
        self.dessiner_plateau()
        self.canvas.update()

        if self.plateau.victoire == Couleur.RED:
            # Message de victoire
            messagebox.showinfo("Information", "Victoire rouge !")
            # Fermer la fenêtre principale
            root.destroy()

        elif self.plateau.victoire == Couleur.BLACK:
            # Message de victoire
            messagebox.showinfo("Information", "Victoire noire !")
            # Fermer la fenêtre principale
            root.destroy()

        elif self.plateau.victoire == Couleur.GREY:
            # Message de victoire
            messagebox.showinfo("Information", "Nulle...")
            # Fermer la fenêtre principale
            root.destroy()

        elif self.J1 == "humain" or self.J2 == "humain":
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.bind("<ButtonRelease-1>", self.on_drag_stop)

        else:
            if self.plateau.tour_actuel == Couleur.RED:
                self.coup_IA(modele=self.modele_rouge)
            else:
                self.coup_IA(modele=self.modele_noir)

    #Lors du tour de l'IA, on ne fait pas déplacer les pieces
    def on_drag_stop_noclick(self, event):
        #Arreter de déplacer la pièce
        self.drag_data["item"] = None
        self.dessiner_plateau()


# Lancement de l'application
if __name__ == "__main__":
    root = tk.Tk()
    # Vous pouvez modifier les dimensions ici : InterfaceEchecs(root, lignes=10, colonnes=10)
    app = Interface_xiangqi(root, Joueur1='humain', Joueur2='Aleatoire')
    root.mainloop()