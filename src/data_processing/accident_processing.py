import pandas as pd
import glob
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

dossier = BASE_DIR / "data" / "raw" / "maritime_accidents"
fichiers = glob.glob(os.path.join(dossier, '*.csv'))

def get_time(heure:float)->str:
    if pd.isna(heure) or heure == "":
        return None
    return str(heure).split(":")[0]

def get_saison(date):
    m, d = date.month, date.day
    if (m == 3 and d >= 20) or (3 < m < 6) or (m == 6 and d < 21):
        return 'Spring'
    elif (m == 6 and d >= 21) or (6 < m < 9) or (m == 9 and d < 22):
        return 'Summer'
    elif (m == 9 and d >= 22) or (9 < m < 12) or (m == 12 and d < 21):
        return 'Autumn'
    else:
        return 'Winter'

CAUSES_HUMAINES = [
    "Body movement under or with physical stress (generally leading to an internal injury)",
    "Body movement without any physical stress (generally leading to an external injury)",
    "Breakage; bursting; splitting; slipping; fall; collapse of Material Agent",
    "Deviation by overflow; overturn; leak; flow; vaporisation; emission",
    "Deviation due to electrical problems; explosion; fire",
    "Loss of control (total or partial) of machine; means of transport or handling equipment; handheld tool; object; animal",
    "Shock; fright; violence; aggression; threat; presence",
    "Slipping - Stumbling and falling - Fall of persons",
]

def get_cause_humaine(deviation):
    """Ramène une valeur de 'Deviation (CE)' à sa catégorie principale, en gérant aussi les
    valeurs tronquées dans les données sources (ex: 'Loss of control (total or partial) of m')."""
    if pd.isna(deviation):
        return None
    deviation = str(deviation)
    for cause in CAUSES_HUMAINES:
        if deviation.startswith(cause) or cause.startswith(deviation):
            return cause
    return deviation


def dms_to_decimal(coord):
    """
    Convertit une coordonnée au bon format
    """
    if pd.isna(coord):
        return None
 
    coord = str(coord).strip()
    if coord == "":
        return None
    try:
        return float(coord)
    except ValueError:
        pass
    match = re.match(
        r"(\d+(?:\.\d+)?)\s*[°]\s*(\d+(?:\.\d+)?)\s*['′]?\s*(?:(\d+(?:\.\d+)?)\s*[\"″])?\s*([NSEWnsew])",
        coord
    )
    if not match:
        return None
 
    deg, minutes, secondes, direction = match.groups()
    deg = float(deg)
    minutes = float(minutes)
    secondes = float(secondes) if secondes else 0.0
 
    decimal = deg + minutes / 60 + secondes / 3600
 
    if direction.upper() in ("S", "W"):
        decimal = -decimal
 
    return decimal

def charger_donnees():
    total = pd.DataFrame()

    for f in fichiers:
        df = pd.read_csv(f)
        df['gravite'] = df['Occurrence severity'] # Gravité de l'accident
        df['long'] = df['Longitude'].apply(dms_to_decimal) # Longitude convertie en degrés décimaux
        df['lat'] = df['Latitude'].apply(dms_to_decimal) # Latitude convertie en degrés décimaux
        df['heure'] = df['Time (LT) of occurrence'].apply(get_time) # Heure de l'accident, sans les minutes (ex 14 pour 14h32)
        df = df.dropna(subset=["heure"])
        df['port'] = df['Port of accident'] # Port où a eu lieu l'accident
        df['Date of occurrence'] = pd.to_datetime(df['Date of occurrence'], errors='coerce') # Date de l'accident, convertie en datetime
        df = df.dropna(subset=["Date of occurrence"])
        df['saison'] = df['Date of occurrence'].apply(get_saison) # Saison durant laquelle a eu lieu l'accident
        df['annee'] = df['Date of occurrence'].dt.year # Année de l'accident
        df['bateau'] = df['Ship / craft type'].str.split(' - ').str[0] # Type de bateau 
        df = df.dropna(subset=["bateau"])
        df = df[df["bateau"] != "Unknown"]
        df['type_accident'] = df['Occurrence with ship(s)'].str.split(' - ').str[0] # Type d'accident 
        df['cause_accident_humain'] = df['Deviation (CE)'].apply(get_cause_humaine) # Cause précise d'un accident humain

        total = pd.concat([total, df], ignore_index=True)
    print(f"Total des enregistrements: {len(total)}")
    return total

def charger_donnees_sans_doublons():
    df = charger_donnees()
    
    colonnes_doublons = [
        'Date of occurrence',
        'Ship / craft type', 
        'Latitude',
        'Longitude',
        'Time (LT) of occurrence',
        'Port of accident'
    ]
    
    df_sans_doublons = df.drop_duplicates(subset=colonnes_doublons, keep='first')
    
    return df_sans_doublons

df_final = charger_donnees_sans_doublons()
df_final.to_csv(PROCESSED_DIR / "maritime_accidents.csv", index=False)