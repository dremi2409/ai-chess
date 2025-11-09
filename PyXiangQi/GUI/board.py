import sys
from pathlib import Path

# Ajoute le dossier parent 'mon_projet' au PYTHONPATH
projet_root = Path(__file__).parent.parent.parent
print(projet_root)
sys.path.insert(0, str(projet_root))

import tkinter as tk
from PIL import Image, ImageTk
import os
from itertools import chain
from tkinter import messagebox
from PyXiangQi.Gameplay.plt import Plateau, Couleur

image_path = "PyXiangQi\\GUI\\pieces_canvas"


class Interface_xiangqi:
    #Interface graphique du XiangQi
    def __init__(self, root: tk.Tk, lignes: int = 10, colonnes: int = 9, title = "让游戏开始!", 
                 taille_case = 70, Plt=None, position='DROIT'):
        self.root = root

        self.lignes = lignes
        self.position = position
        self.colonnes = colonnes
        self.root.title(title)
        self.taille_case = taille_case

        # Variables to track dragging
        self.drag_data = {"x": 0, "y": 0, "item": None, "col":0, "lin":0}

        self.img_ref = []

        if Plt:
            self.plateau = Plt
            
        else:
            self.plateau = Plateau(self.lignes, self.colonnes)

        self.canvas = tk.Canvas(root, width=colonnes * self.taille_case, 
                                height=lignes * self.taille_case)
        self.canvas.pack()

        # Variables pour le drag & drop
        self.piece_selectionnee = None
        self.position_depart = None
        self.objet_drag = None

        # Événements
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_stop)
        
        self.dessiner_plateau()

    def dessiner_plateau(self):
        """Dessine le plateau et les pièces"""
        self.canvas.delete("all")
        
        # Dessiner les cases
        for ligne in range(self.lignes):
            for col in range(self.colonnes):
                x1 = col * self.taille_case
                y1 = ligne * self.taille_case
                x2 = x1 + self.taille_case
                y2 = y1 + self.taille_case
                
                couleur = "#B58863"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=couleur, outline="")

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
            for col in range(self.colonnes):
                if self.position == 'DROIT': 
                    piece = self.plateau.obtenir_piece(col, ligne)

                else:
                    piece = self.plateau.obtenir_piece_inverse(col, ligne)

                if piece != "-":
                    image = ImageTk.PhotoImage(Image.open(os.path.join(image_path,piece.symbole()+".png")).resize((60,60)))
                    x = (col * self.taille_case + self.taille_case // 2)
                    y = (ligne * self.taille_case + self.taille_case // 2)
                        
                    # Add the image to the Canvas
                    self.canvas.create_image(x, y, image=image)  # Position at (200, 200)

                    # Keep a reference to the image to prevent garbage collection
                    self.img_ref.append(image)

    def dessiner_coups_possibles(self, col, ligne):
        for coup_possible in self.plateau.grille_coups[ligne][col]:
            image = ImageTk.PhotoImage(Image.open(os.path.join(image_path,"spot.png")).resize((20,20)))
            x = (int(coup_possible[1]) * self.taille_case + self.taille_case // 2)
            y = (int(coup_possible[0]) * self.taille_case + self.taille_case // 2)
                        
            # Add the image to the Canvas
            self.canvas.create_image(x, y, image=image)  # Position at (200, 200)

            # Keep a reference to the image to prevent garbage collection
            self.img_ref.append(image)


    # Mouse events for dragging
    def on_drag_start(self, event):
        """Gère le clic initial pour sélectionner une pièce"""
        col = event.x // self.taille_case
        ligne = event.y // self.taille_case

        #Enregistrer la position de base pour connaître les coups possibles
        self.drag_data["col"] = col
        self.drag_data["lin"] = ligne
        
        if self.position == 'DROIT': 
            piece = self.plateau.obtenir_piece(col, ligne)

        else:
            piece = self.plateau.obtenir_piece_inverse(col, ligne)

        self.drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        item_type = self.canvas.type(self.drag_data["item"])

        if piece !="-" and piece and item_type == 'image':
            self.piece_selectionnee = piece
            if self.plateau.grille_coups[ligne][col]:
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

        if self.plateau.grille_coups[self.drag_data["lin"]][self.drag_data["col"]] and self.drag_data["item"] != None and 0 <= col < self.colonnes and 0 <= ligne < self.lignes :
            if str(ligne)+str(col) in self.plateau.grille_coups[self.drag_data["lin"]][self.drag_data["col"]]:
                self.plateau.deplacer_piece(self.drag_data["col"],self.drag_data["lin"],col,ligne)
                if self.plateau.victoire == Couleur.RED:
                    # Message de victoire
                    messagebox.showinfo("Information", "Victoire rouge !")
                    # Fermer la fenêtre principale
                    root.destroy()

                if self.plateau.victoire == Couleur.BLACK:
                    # Message de victoire
                    messagebox.showinfo("Information", "Victoire noire !")
                    # Fermer la fenêtre principale
                    root.destroy()

        self.dessiner_plateau()

        #Arreter de déplacer la pièce
        self.drag_data["item"] = None



# Lancement de l'application
if __name__ == "__main__":
    while True:
        try:
            root = tk.Tk()
            # Vous pouvez modifier les dimensions ici : InterfaceEchecs(root, lignes=10, colonnes=10)
            app = Interface_xiangqi(root)
            root.mainloop()

        except:
            print("La partie est finie")

