import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_processing.loader import charger_flotte
import matplotlib.pyplot as plt

def flotte_totale_par_annee(economie="Monde"):
    """Nombre total de navires (tous types confondus) par année"""
    flotte = charger_flotte(economie)
    return flotte[flotte['type_navire'] == 'Flotte totale'].set_index('annee')['nombre_navires'].sort_index()


def flotte_par_type_et_annee(economie="Monde"):
    """Nombre de navires par type, pour chaque année (une colonne par type, hors 'Flotte totale')"""
    flotte = charger_flotte(economie)
    flotte = flotte[flotte['type_navire'] != 'Flotte totale']
    return flotte.pivot(index='annee', columns='type_navire', values='nombre_navires')


def graphique_flotte_totale(comptage, economie="Monde"):
    plt.figure(figsize=(12, 6))
    plt.plot(comptage.index, comptage.values, marker="o")
    plt.title(f"Taille de la flotte marchande ({economie}) par année", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Nombre de navires", fontsize=12)
    plt.grid(True)
    os.makedirs("visualization/fleet", exist_ok=True)
    plt.tight_layout()
    plt.savefig("visualization/fleet/graph_global_fleet.png", dpi=150)
    plt.close()


def graphique_flotte_par_type(comptage, economie="Monde"):
    plt.figure(figsize=(12, 6))
    for type_navire in comptage.columns:
        plt.plot(comptage.index, comptage[type_navire], marker="o", label=type_navire)
    plt.title(f"Flotte marchande par type de navire ({economie})", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Nombre de navires", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("visualization/fleet", exist_ok=True)
    plt.tight_layout()
    plt.savefig("visualization/fleet/graph_fleet_by_economy.png", dpi=150)
    plt.close()


graphique_flotte_totale(flotte_totale_par_annee())
graphique_flotte_par_type(flotte_par_type_et_annee())