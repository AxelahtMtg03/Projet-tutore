import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'traitement_donnee'))
from charge import charger_donnees_finales

import matplotlib.pyplot as plt

def accidents_par_annee():
    total = charger_donnees_finales()
    comptage = total.groupby("annee").size()
    return comptage

def graphique_annee(total):
    plt.figure(figsize=(12,6))

    plt.plot(total.index,total.values,marker="o"
    )

    plt.title(f"Nombre d'accidents par année", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True)

    plt.show()

graphique_annee(accidents_par_annee())