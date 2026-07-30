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

#reconstruire_flotte_csv("données_flotte_global/US.MerchantFleet_20260724_003208.csv")


def charger_flotte(economie="Monde"):

    flotte = pd.read_csv("data/flotte_par_pays.csv")
    return flotte[flotte['economie'] == economie]