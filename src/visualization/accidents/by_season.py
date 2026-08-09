import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_processing.loader import charger_donnees_finales

def saison_global():
    # Fusionner tous les fichiers
    total = charger_donnees_finales()
    comptage_total = total['saison'].value_counts()
    ordre_saisons = ['Spring', 'Summer', 'Autumn', 'Winter']  # doit matcher get_saison() (accidents_processing.py)
    comptage_total = comptage_total.reindex(ordre_saisons)
    return comptage_total

def grapique_saison_global(total):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(total.index, total.values, 
                color=['green', 'red', 'orange', 'blue'])
    plt.title('Nombre d\'accidents par saison', fontsize=14)
    plt.xlabel('Saison', fontsize=12)
    plt.ylabel('Nombre d\'accidents', fontsize=12)
    plt.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    os.makedirs("visualization/accidents", exist_ok=True)
    plt.tight_layout()
    plt.savefig("visualization/accidents/accidents_by_season.png", dpi=150)
    plt.show()

def accidents_par_annee_pour_saison(saison:str):
    """Nombre d'accidents par année, pour UNE SEULE saison filtrée (ex: 'Spring')"""
    total = charger_donnees_finales()
    total = total[total["saison"] == saison]
    comptage = total.groupby("annee").size()
    return comptage

def graphique_saison_annee_une_saison(comptage, saison):
    """Affiche l'évolution du nombre d'accidents par année, pour une seule saison"""
    plt.figure(figsize=(12,6))

    plt.plot(comptage.index, comptage.values, marker="o")

    plt.title(f"Nombre d'accidents en {saison} par année", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True)

    os.makedirs("visualization/accidents", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"visualization/accidents/accidents_by_year_{saison}.png", dpi=150)
    plt.close()

def accidents_par_annee_toutes_saisons():
    """Nombre d'accidents par année, pour TOUTES les saisons (une colonne par saison)"""
    total = charger_donnees_finales()
    comptage = total.groupby(["annee","saison"]).size().unstack(fill_value=0)
    return comptage

def graphique_saison_annee_toutes_saisons(comptage):
    """Affiche l'évolution du nombre d'accidents par année, une courbe par saison"""
    plt.figure(figsize=(12,6))

    for saison in comptage.columns:
        plt.plot(
            comptage.index,
            comptage[saison],
            marker="o",
            label=saison
        )

    plt.title("Nombre d'accidents par saison et par année")
    plt.xlabel("Année")
    plt.ylabel("Nombre d'accidents")

    plt.grid(True)
    plt.legend()

    os.makedirs("visualization/accidents", exist_ok=True)
    plt.tight_layout()
    plt.savefig("visualization/accidents/accidents_by_season_per_year.png", dpi=150)
    plt.close()

# grapique_saison_global(saison_global())

# Une seule saison :
saison = "Summer"
graphique_saison_annee_une_saison(accidents_par_annee_pour_saison(saison), saison)

# Toutes les saisons sur le même graphique :
graphique_saison_annee_toutes_saisons(accidents_par_annee_toutes_saisons())