import pandas as pd
import glob
import os
import re
import numpy as np
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import cKDTree

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


COULEURS = {
    'Forte diminution': '#08306b',
    'Faible diminution': '#27ae60',
    'Stable': '#8e44ad',
    'Faible augmentation': '#f4d03f',
    'Forte augmentation': '#e74c3c',
    'Données insuffisantes': '#cccccc',
}
 
 
def categoriser_tendance(ref, valeur):
    """Classe une case (référence historique vs valeur réelle) en 5 catégories,
    + 'Données insuffisantes' quand la référence est trop faible pour être fiable."""
    if ref < 0.5:
        return 'Données insuffisantes'
 
    ecart_absolu = valeur - ref
    if abs(ecart_absolu) < 1.5:
        return 'Stable'
 
    pct = ecart_absolu / ref * 100
    if pct <= -30:
        return 'Forte diminution'
    elif pct <= -10:
        return 'Faible diminution'
    elif pct < 10:
        return 'Stable'
    elif pct < 30:
        return 'Faible augmentation'
    else:
        return 'Forte augmentation'
 
 
def calculer_tendance_reelle(chemin_accidents="data/processed/maritime_accidents.csv",
                              chemin_density="data/processed/vessel_density.csv",
                              taille_grille_deg=1.5,
                              distance_max_m=50_000,
                              chemin_sortie="data/processed/real_trend_zones.csv",
                              sauvegarder=True):
    """Calcule, pour chaque case, la tendance RÉELLE des accidents (valeur de l'année
    vs moyenne historique de la zone) — sans aucune prédiction.
 
    IMPORTANT : les zones utilisées sont les cases NATIVES de vessel_density (comme
    dans le modèle de prédiction), pas une grille directe sur les coordonnées des
    accidents. C'est ce qui garantit que 'référence historique' soit identique entre
    cette carte et la carte de prédiction, pour une même zone.
 
    Retourne le DataFrame agrégé (une ligne par case x année), et le sauvegarde en
    CSV si sauvegarder=True.
    """
 
    # --------------------------------------------------
    # 1. Chargement
    # --------------------------------------------------
    accidents = pd.read_csv(chemin_accidents)
    density = pd.read_csv(chemin_density)
    accidents = accidents.dropna(subset=['lat', 'long', 'annee'])
 
    # --------------------------------------------------
    # 2. Reprojection des accidents (degrés WGS84) vers la grille de densité (mètres, EPSG:3035)
    # --------------------------------------------------
    transformer_vers_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_acc, y_acc = transformer_vers_3035.transform(accidents['long'].values, accidents['lat'].values)
    accidents['x'] = x_acc
    accidents['y'] = y_acc
 
    # --------------------------------------------------
    # 3. Zones natives = cases de vessel_density (identique au modèle de prédiction)
    # --------------------------------------------------
    zones = density[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    zones['zone_id'] = zones.index
    zone_tree = cKDTree(zones[['latitude', 'longitude']].values)
 
    # --------------------------------------------------
    # 4. Assignation de chaque accident à la case de densité la plus proche
    # --------------------------------------------------
    distances, nearest_zone_idx = zone_tree.query(accidents[['y', 'x']].values)
    accidents['zone_id'] = zones.loc[nearest_zone_idx, 'zone_id'].values
    accidents['distance_zone'] = distances
 
    nb_avant = len(accidents)
    accidents = accidents[accidents['distance_zone'] <= distance_max_m].copy()
    print(f"{len(accidents)}/{nb_avant} accidents dans la zone de couverture de vessel_density")
 
    # --------------------------------------------------
    # 5. Référence historique par zone (moyenne annuelle réelle, toute la période)
    # --------------------------------------------------
    annee_min = int(accidents['annee'].min())
    annee_max = int(accidents['annee'].max())
    nb_annees_total = annee_max - annee_min + 1
 
    accidents_par_zone_annee = accidents.groupby(['zone_id', 'annee']).size().reset_index(name='nb_accidents')
 
    reference_zone = (
        accidents_par_zone_annee.groupby('zone_id')['nb_accidents'].sum() / nb_annees_total
    ).reset_index(name='reference_historique')
    reference_zone = zones[['zone_id']].merge(reference_zone, on='zone_id', how='left')
    reference_zone['reference_historique'] = reference_zone['reference_historique'].fillna(0)
 
    # --------------------------------------------------
    # 6. Table complète : chaque zone x chaque année réellement connue
    # --------------------------------------------------
    toutes_annees = pd.DataFrame({'annee': range(annee_min, annee_max + 1)})
    grille_zones_annees = zones[['zone_id']].merge(toutes_annees, how='cross')
 
    resultats = grille_zones_annees.merge(accidents_par_zone_annee, on=['zone_id', 'annee'], how='left')
    resultats['nb_accidents'] = resultats['nb_accidents'].fillna(0)
    resultats = resultats.merge(reference_zone, on='zone_id', how='left')
 
    # --------------------------------------------------
    # 7. Reprojection des centroïdes de zones natives en degrés (WGS84)
    # --------------------------------------------------
    transformer_vers_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = transformer_vers_4326.transform(zones['longitude'].values, zones['latitude'].values)
    zones['lat_deg'] = lat_deg
    zones['lon_deg'] = lon_deg
 
    resultats = resultats.merge(zones[['zone_id', 'lat_deg', 'lon_deg']], on='zone_id', how='left')
 
    # --------------------------------------------------
    # 8. Agrégation dans une grille droite en degrés (même taille que la prédiction)
    # --------------------------------------------------
    resultats['grille_lat'] = np.floor(resultats['lat_deg'] / taille_grille_deg) * taille_grille_deg
    resultats['grille_lon'] = np.floor(resultats['lon_deg'] / taille_grille_deg) * taille_grille_deg
 
    agrege = resultats.groupby(['grille_lat', 'grille_lon', 'annee']).agg(
        nb_accidents=('nb_accidents', 'sum'),
        reference_historique=('reference_historique', 'sum'),
        nb_zones=('zone_id', 'nunique'),
    ).reset_index()
 
    # --------------------------------------------------
    # 9. Classification en 5 catégories (+ "Données insuffisantes")
    # --------------------------------------------------
    agrege['pct_evolution'] = np.where(
        agrege['reference_historique'] >= 0.5,
        (agrege['nb_accidents'] - agrege['reference_historique']) / agrege['reference_historique'] * 100,
        np.nan
    )
    agrege['categorie'] = agrege.apply(
        lambda r: categoriser_tendance(r['reference_historique'], r['nb_accidents']), axis=1
    )
    agrege['couleur'] = agrege['categorie'].map(COULEURS)
 
    print(f"Données de {annee_min} à {annee_max} ({nb_annees_total} années) — {len(agrege)} lignes (cases x années)")
    print("Répartition par catégorie :")
    print(agrege['categorie'].value_counts())
 
    if sauvegarder:
        os.makedirs(os.path.dirname(chemin_sortie), exist_ok=True)
        agrege.to_csv(chemin_sortie, index=False)
        print(f"Résultats sauvegardés dans {chemin_sortie}")

    return agrege



# df_final = charger_donnees_sans_doublons()
# df_final.to_csv(PROCESSED_DIR / "maritime_accidents.csv", index=False)

# calculer_tendance_reelle()