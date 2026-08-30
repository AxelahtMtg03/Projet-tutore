# src/prediction/random_forest.py
import os
import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "visualization" / "predictions"

# ============================================================
# COULEURS
# ============================================================

COULEURS = {
    'Forte diminution': '#08306b',
    'Faible diminution': '#27ae60',
    'Stable': '#8e44ad',
    'Faible augmentation': '#f4d03f',
    'Forte augmentation': '#e74c3c',
    'Donnees insuffisantes': '#cccccc',
}


# ============================================================
# FONCTION 1 : MODELE REGRESSION (ORIGINAL)
# ============================================================

def run_regression_prediction():
    """
    Modele Random Forest REGRESSION original.
    Predit le nombre exact d'accidents par zone et annee.
    """
    
    print("="*60)
    print("RANDOM FOREST - REGRESSION (ORIGINAL)")
    print("="*60)
    
    # 1. Chargement des donnees
    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")
    
    accidents = accidents.dropna(subset=['lat', 'long', 'annee'])
    density['annee'] = pd.to_datetime(density['time']).dt.year
    
    # 2. Reprojection des accidents
    transformer_vers_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_acc, y_acc = transformer_vers_3035.transform(accidents['long'].values, accidents['lat'].values)
    accidents['x'] = x_acc
    accidents['y'] = y_acc
    
    # 3. Zones natives = cases de vessel_density
    zones = density[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    zones['zone_id'] = zones.index
    zone_tree = cKDTree(zones[['latitude', 'longitude']].values)
    
    # 4. Assignation des accidents aux zones
    DISTANCE_MAX_M = 50_000
    distances, nearest_zone_idx = zone_tree.query(accidents[['y', 'x']].values)
    accidents['zone_id'] = zones.loc[nearest_zone_idx, 'zone_id'].values
    accidents['distance_zone'] = distances
    
    nb_avant = len(accidents)
    accidents = accidents[accidents['distance_zone'] <= DISTANCE_MAX_M].copy()
    print(f"{len(accidents)}/{nb_avant} accidents dans la zone de couverture")
    
    # 5. Reference historique
    ANNEE_MIN_ACC = int(accidents['annee'].min())
    ANNEE_MAX_ACC = int(accidents['annee'].max())
    NB_ANNEES_TOTAL = ANNEE_MAX_ACC - ANNEE_MIN_ACC + 1
    print(f"Donnees disponibles jusqu'en {ANNEE_MAX_ACC}")
    
    accidents_par_zone_annee = (
        accidents.groupby(['zone_id', 'annee']).size().reset_index(name='nb_accidents')
    )
    
    reference_zone = (
        accidents_par_zone_annee.groupby('zone_id')['nb_accidents'].sum() / NB_ANNEES_TOTAL
    ).reset_index(name='reference_historique')
    reference_zone = zones[['zone_id']].merge(reference_zone, on='zone_id', how='left')
    reference_zone['reference_historique'] = reference_zone['reference_historique'].fillna(0)
    
    # 6. Statistiques de densite
    density = density.merge(zones[['latitude', 'longitude', 'zone_id']], on=['latitude', 'longitude'], how='left')
    
    density_stats = density.groupby(['zone_id', 'annee']).agg(
        densite_moyenne=('vd', 'mean'),
        densite_mediane=('vd', 'median'),
        densite_max=('vd', 'max'),
        densite_std=('vd', 'std'),
        nb_mesures=('vd', 'count')
    ).reset_index()
    density_stats['densite_std'] = density_stats['densite_std'].fillna(0)
    
    density_stats = density_stats.merge(reference_zone, on='zone_id', how='left')
    
    train_data = density_stats.merge(accidents_par_zone_annee, on=['zone_id', 'annee'], how='left')
    train_data['nb_accidents'] = train_data['nb_accidents'].fillna(0)
    
    # 7. Entrainement
    FEATURES = ['densite_moyenne', 'densite_mediane', 'densite_max', 'densite_std', 'nb_mesures',
                'reference_historique']
    
    model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(train_data[FEATURES], train_data['nb_accidents'])
    print(f"Modele entraine sur {len(train_data)} lignes")
    
    # 8. Projection de la densite
    ANNEES_CIBLE = list(range(2023, 2031))
    ANNEES_CONNUES_DENSITE = sorted(density_stats['annee'].unique())
    
    x_annees = np.array(ANNEES_CONNUES_DENSITE, dtype=float)
    x_mean = x_annees.mean()
    x_var = np.sum((x_annees - x_mean) ** 2)
    
    pentes = {}
    intercepts = {}
    bornes_min = {}
    bornes_max = {}
    for feature in FEATURES:
        if feature == 'reference_historique':
            continue
        pivot = density_stats.pivot(index='zone_id', columns='annee', values=feature)
        pivot = pivot.reindex(columns=ANNEES_CONNUES_DENSITE)
        y = pivot.values
        y_mean = np.nanmean(y, axis=1, keepdims=True)
        pente = np.nansum((x_annees - x_mean) * (y - y_mean), axis=1) / x_var
        intercept = y_mean.flatten() - pente * x_mean
        pentes[feature] = pd.Series(pente, index=pivot.index)
        intercepts[feature] = pd.Series(intercept, index=pivot.index)
        bornes_min[feature] = pd.Series(np.nanmin(y, axis=1), index=pivot.index)
        bornes_max[feature] = pd.Series(np.nanmax(y, axis=1), index=pivot.index)
    
    def projeter_densite(annee_cible):
        X_projete = pd.DataFrame({'zone_id': zones['zone_id']})
        for feature in FEATURES:
            if feature == 'reference_historique':
                continue
            valeurs_brutes = intercepts[feature].reindex(zones['zone_id']).values + \
                               pentes[feature].reindex(zones['zone_id']).values * annee_cible
            mini = bornes_min[feature].reindex(zones['zone_id']).values
            maxi = bornes_max[feature].reindex(zones['zone_id']).values
            valeurs_plafonnees = np.clip(valeurs_brutes, a_min=mini, a_max=maxi)
            X_projete[feature] = np.clip(valeurs_plafonnees, a_min=0, a_max=None)
        X_projete = X_projete.merge(reference_zone, on='zone_id', how='left')
        return X_projete
    
    # 9. Predictions
    lignes = []
    for annee in ANNEES_CIBLE:
        X_projete = projeter_densite(annee)
        valeurs = X_projete[['zone_id']].copy()
        valeurs['valeur_predite'] = model.predict(X_projete[FEATURES])
        
        if annee <= ANNEE_MAX_ACC:
            reel_annee = accidents_par_zone_annee[accidents_par_zone_annee['annee'] == annee]
            valeurs = valeurs.merge(reel_annee[['zone_id', 'nb_accidents']], on='zone_id', how='left')
            valeurs['valeur_reelle'] = valeurs['nb_accidents'].fillna(0)
            valeurs = valeurs.drop(columns='nb_accidents')
        else:
            valeurs['valeur_reelle'] = np.nan
        
        valeurs['annee'] = annee
        lignes.append(valeurs)
    
    resultats_annuels = pd.concat(lignes, ignore_index=True)
    resultats_annuels = resultats_annuels.merge(reference_zone, on='zone_id', how='left')
    
    # 10. Reprojection en degres
    transformer_vers_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = transformer_vers_4326.transform(zones['longitude'].values, zones['latitude'].values)
    zones['lat_deg'] = lat_deg
    zones['lon_deg'] = lon_deg
    
    resultats_annuels = resultats_annuels.merge(zones[['zone_id', 'lat_deg', 'lon_deg']], on='zone_id', how='left')
    
    # 11. Agregation en grille
    TAILLE_GRILLE_DEG = 1.5
    resultats_annuels['grille_lat'] = np.floor(resultats_annuels['lat_deg'] / TAILLE_GRILLE_DEG) * TAILLE_GRILLE_DEG
    resultats_annuels['grille_lon'] = np.floor(resultats_annuels['lon_deg'] / TAILLE_GRILLE_DEG) * TAILLE_GRILLE_DEG
    
    agrege = resultats_annuels.groupby(['grille_lat', 'grille_lon', 'annee']).agg(
        predit_total=('valeur_predite', 'sum'),
        reel_total=('valeur_reelle', lambda s: np.nan if s.isna().all() else s.sum()),
        reference_totale=('reference_historique', 'sum'),
        nb_zones=('zone_id', 'nunique'),
    ).reset_index()
    
    # 12. Categorisation
    def categoriser(ref, valeur):
        if ref < 0.5:
            return 'Donnees insuffisantes'
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
    
    agrege['pct_evolution'] = np.where(
        agrege['reference_totale'] >= 0.5,
        (agrege['predit_total'] - agrege['reference_totale']) / agrege['reference_totale'] * 100,
        np.nan
    )
    agrege['categorie'] = agrege.apply(lambda r: categoriser(r['reference_totale'], r['predit_total']), axis=1)
    agrege['couleur'] = agrege['categorie'].map(COULEURS)
    agrege['predit_affiche'] = np.floor(agrege['predit_total']).astype(int)
    
    # 13. Sauvegarde
    os.makedirs(DATA_PATH, exist_ok=True)
    agrege.to_csv(DATA_PATH / "predictions_random_forest_densite_2023_2030.csv", index=False)
    print(f"Donnees sauvegardees dans {DATA_PATH / 'predictions_random_forest_densite_2023_2030.csv'}")
    
    # 14. Evaluation
    evaluation = agrege[agrege['reel_total'].notna()].copy()
    if len(evaluation) > 0:
        mae = mean_absolute_error(evaluation['reel_total'], evaluation['predit_total'])
        rmse = np.sqrt(mean_squared_error(evaluation['reel_total'], evaluation['predit_total']))
        print(f"MAE: {mae:.2f} accidents")
        print(f"RMSE: {rmse:.2f} accidents")
        for annee, groupe in evaluation.groupby('annee'):
            mae_annee = mean_absolute_error(groupe['reel_total'], groupe['predit_total'])
            print(f"  {int(annee)}: MAE = {mae_annee:.2f} accidents ({len(groupe)} cases)")
    
    return agrege


# 1. Modele regression (original)
# print("1. MODELE REGRESSION (original)")
# print("-"*40)
# agrege = run_regression_prediction()