#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:20:20 2024

@author: cleohaddad
"""

import tkinter as tk
from tkinter import filedialog, messagebox

# Remplacez ces fonctions par vos implémentations
#from votre_module import pretraitement, detection  # Assurez-vous que le nom du fichier est correct

# Fonction pour charger un fichier CSV
def charger_fichier():
    fichier = filedialog.askopenfilename(
        title="Sélectionnez un fichier CSV",
        filetypes=[("Fichiers CSV", "*.csv")],
    )
    if fichier:
        fichier_path.set(fichier)
        fichier_label.config(text=f"Fichier sélectionné : {fichier}")
    else:
        messagebox.showerror("Erreur", "Aucun fichier sélectionné.")

# Fonction principale
def traiter_fichier():
    fichier = fichier_path.get()
    if not fichier:
        messagebox.showerror("Erreur", "Veuillez sélectionner un fichier.")
        return

    try:
        if pretraitement_var.get():  # Si l'utilisateur souhaite un prétraitement
            fichier_pretraite = pretraitement(fichier)  # Appel à votre fonction
            anomalies = detection(fichier_pretraite)  # Appel à votre fonction
            resultats["text"] = (
                f"Fichier prétraité avec succès : {fichier_pretraite}\n"
                f"Anomalies détectées :\n{anomalies}"
            )
        else:  # Si l'utilisateur ne souhaite pas de prétraitement
            anomalies = detection(fichier)  # Appel à votre fonction
            resultats["text"] = f"Anomalies détectées :\n{anomalies}"
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

# Configuration de l'interface graphique
root = tk.Tk()
root.title("Analyse de Données CSV")

# Variables
fichier_path = tk.StringVar()
pretraitement_var = tk.BooleanVar(value=False)

# Widgets
titre = tk.Label(root, text="Analyse de Données CSV", font=("Helvetica", 16))
titre.pack(pady=10)

btn_charger = tk.Button(root, text="Charger un fichier CSV", command=charger_fichier)
btn_charger.pack()

fichier_label = tk.Label(root, text="Aucun fichier sélectionné", fg="gray")
fichier_label.pack(pady=5)

pretraitement_checkbox = tk.Checkbutton(root, text="Effectuer un prétraitement", variable=pretraitement_var)
pretraitement_checkbox.pack()

btn_traiter = tk.Button(root, text="Traiter", command=traiter_fichier)
btn_traiter.pack(pady=10)

resultats = tk.Label(root, text="", wraplength=500, justify="left", fg="blue")
resultats.pack(pady=10)


root.mainloop()

