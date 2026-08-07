import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def charger_donnees_finales():
    """Charge le CSV final déjà nettoyé (sans doublons) ; à utiliser dans les autres fichiers du projet"""
    df = pd.read_csv(PROCESSED_DIR / "maritime_accidents.csv")
    return df

def charger_flotte(economie="Monde"):
    flotte = pd.read_csv(PROCESSED_DIR / "global_fleet.csv")
    return flotte[flotte['economie'] == economie]

def charger_densite():
    df = pd.read_csv(PROCESSED_DIR / "vessel_density.csv")
    return df