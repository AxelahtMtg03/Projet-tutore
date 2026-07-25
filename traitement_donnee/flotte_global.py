import matplotlib.pyplot as plt
import pandas as pd
import re

def reconstruire_flotte_csv(chemin_export_brut, chemin_sortie="data/flotte_par_pays.csv"):
    """Transforme un export UNCTADstat brut (format large) en CSV propre (format long)."""
    df = pd.read_csv(chemin_export_brut)
    colonnes_value = [c for c in df.columns if c.endswith('_Nombre_de_navires_Value')]

    lignes = []
    for _, row in df.iterrows():
        economie = row['Economy_Label']
        for col in colonnes_value:
            m = re.match(r"(\d{4})_(.+)_Nombre_de_navires_Value", col)
            if not m:
                continue
            annee, type_navire = m.groups()
            valeur = row[col]
            if pd.notna(valeur):
                lignes.append({
                    'economie': economie,
                    'annee': int(annee),
                    'type_navire': type_navire,
                    'nombre_navires': int(valeur)
                })

    flotte = pd.DataFrame(lignes)
    flotte.to_csv(chemin_sortie, index=False)
    print(f"{len(flotte)} lignes écrites dans {chemin_sortie}")
    return flotte

reconstruire_flotte_csv("data_brut/US.MerchantFleet_20260724_003208.csv")


def charger_flotte(economie="Monde"):

    flotte = pd.read_csv("data/flotte_par_pays.csv")
    return flotte[flotte['economie'] == economie]


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
    plt.tight_layout()
    plt.show()


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
    plt.show()


if __name__ == "__main__":
    graphique_flotte_totale(flotte_totale_par_annee())
    graphique_flotte_par_type(flotte_par_type_et_annee())