import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_processing.loader import charger_donnees_finales

def comptage_croise(colonne_x:str, colonne_serie:str):
    """Croise deux colonnes (ex: saison/bateau) et compte le nombre d'accidents pour chaque combinaison"""
    total = charger_donnees_finales()
    total_europe = total[
        (total["lat"].between(35, 70)) &   # Latitude Europe
        (total["long"].between(-10, 40))   # Longitude Europe
    ]
    comptage = total_europe.groupby([colonne_x, colonne_serie]).size().unstack(fill_value=0)
    return comptage

def graphique(comptage, colonne_x:str, colonne_serie:str):

    plt.figure(figsize=(12,6))

    for valeur_serie in comptage.columns:
        plt.plot(
            comptage.index,
            comptage[valeur_serie],
            marker="o",
            label=valeur_serie
        )

    plt.title("Nombre d'accidents par " + colonne_x + " et par " + colonne_serie)
    plt.xlabel(colonne_x)
    plt.ylabel("Nombre d'accidents")

    plt.grid(True)
    plt.legend()

    os.makedirs("visualization/accidents", exist_ok=True)
    plt.tight_layout()
    nom_fichier = f"visualization/accidents/accidents_by_{colonne_x}_and_{colonne_serie}_europe.png"
    plt.savefig(nom_fichier, dpi=150)
    plt.close()
    
def comptage_croise_bis(colonne_filtre:str, colonne_groupe:str, valeur_filtre:str):
    """Comme comptage_croise, mais filtré ex le nb d'accidnet de bateaux de croisière en fonction des saisons"""
    total = charger_donnees_finales()

    total = total[total[colonne_filtre] == valeur_filtre]
    comptage = total.groupby(colonne_groupe).size()
    return comptage
 
def graphique_bis(comptage, colonne_filtre:str, colonne_groupe:str, valeur_filtre:str):
    """Affiche l'évolution du nombre d'accidents selon colonne_groupe, pour la seule valeur valeur_filtre filtrée en amont"""
    plt.figure(figsize=(12,6))
 
    plt.plot(comptage.index, comptage.values, marker="o")
 
    plt.title(f"Nombre d'accidents par {colonne_groupe}, pour {colonne_filtre} = \"{valeur_filtre}\"", fontsize=14)
    plt.xlabel(f"{colonne_groupe}", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True)

    os.makedirs("visualization/accidents", exist_ok=True)
    plt.tight_layout()
    valeur_nettoyee = valeur_filtre.replace(" ", "_")
    nom_fichier = f"visualization/accidents/accidents_by_{colonne_groupe}_for_{colonne_filtre}_{valeur_nettoyee}.png"
    plt.savefig(nom_fichier, dpi=150)
    plt.close()
 
colonne_x = "saison"
colonne_serie = "bateau"
graphique((comptage_croise(colonne_x, colonne_serie)), colonne_x, colonne_serie)
 
# colonne_filtre = "bateau"
# colonne_groupe = "saison"
# type_bateau = "Cargo ship"
# graphique_bis(comptage_croise_bis(colonne_filtre, colonne_groupe, type_bateau), colonne_filtre, colonne_groupe, type_bateau)