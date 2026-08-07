import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "global_fleet"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def reconstruire_flotte_csv(chemin_export_brut, chemin_sortie=None):
    if chemin_sortie is None:
        chemin_sortie = PROCESSED_DIR / "global_fleet.csv"
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

reconstruire_flotte_csv(RAW_DIR / "US.MerchantFleet_20260724_003208.csv")