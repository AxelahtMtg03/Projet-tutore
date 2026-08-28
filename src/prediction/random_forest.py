import os
import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor


accidents = pd.read_csv("data/processed/maritime_accidents.csv")
density = pd.read_csv("data/processed/vessel_density.csv")

accidents = accidents.dropna(subset=['lat', 'long', 'annee'])
density['annee'] = pd.to_datetime(density['time']).dt.year

transformer_vers_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
x_acc, y_acc = transformer_vers_3035.transform(accidents['long'].values, accidents['lat'].values)
accidents['x'] = x_acc
accidents['y'] = y_acc

zones = density[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
zones['zone_id'] = zones.index

zone_tree = cKDTree(zones[['latitude', 'longitude']].values)

DISTANCE_MAX_M = 50_000  # 50 km

distances, nearest_zone_idx = zone_tree.query(accidents[['y', 'x']].values)
accidents['zone_id'] = zones.loc[nearest_zone_idx, 'zone_id'].values
accidents['distance_zone'] = distances

nb_avant = len(accidents)
accidents = accidents[accidents['distance_zone'] <= DISTANCE_MAX_M].copy()
print(f"{len(accidents)}/{nb_avant} accidents dans la zone de couverture de vessel_density")

ANNEE_MIN_ACC = int(accidents['annee'].min())
ANNEE_MAX_ACC = int(accidents['annee'].max())  # dernière année RÉELLEMENT connue
NB_ANNEES_TOTAL = ANNEE_MAX_ACC - ANNEE_MIN_ACC + 1

print(f"Données d'accidents réelles disponibles jusqu'en {ANNEE_MAX_ACC}")

accidents_par_zone_annee = (
    accidents.groupby(['zone_id', 'annee']).size().reset_index(name='nb_accidents')
)

reference_zone = (
    accidents_par_zone_annee.groupby('zone_id')['nb_accidents'].sum() / NB_ANNEES_TOTAL
).reset_index(name='reference_historique')
reference_zone = zones[['zone_id']].merge(reference_zone, on='zone_id', how='left')
reference_zone['reference_historique'] = reference_zone['reference_historique'].fillna(0)

density = density.merge(zones[['latitude', 'longitude', 'zone_id']], on=['latitude', 'longitude'], how='left')

density_stats = density.groupby(['zone_id', 'annee']).agg(
    densite_moyenne=('vd', 'mean'),
    densite_mediane=('vd', 'median'),
    densite_max=('vd', 'max'),
    densite_std=('vd', 'std'),
    nb_mesures=('vd', 'count')
).reset_index()
density_stats['densite_std'] = density_stats['densite_std'].fillna(0)

# On ajoute la référence historique de la zone comme variable d'entrée : sans ça, le
# modèle ne fait aucune différence entre un port très accidentogène (ex: Hambourg)
# et une simple voie de navigation en mer ouverte avec la même densité de trafic.
density_stats = density_stats.merge(reference_zone, on='zone_id', how='left')

train_data = density_stats.merge(accidents_par_zone_annee, on=['zone_id', 'annee'], how='left')
train_data['nb_accidents'] = train_data['nb_accidents'].fillna(0)

FEATURES = ['densite_moyenne', 'densite_mediane', 'densite_max', 'densite_std', 'nb_mesures',
            'reference_historique']

model = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1
)
model.fit(train_data[FEATURES], train_data['nb_accidents'])
print(f"Modèle entraîné sur {len(train_data)} lignes (zones x années)")

# Avant : on utilisait toujours la densité de 2020, donc la prédiction était
# identique pour 2023...2030. Maintenant : on extrapole une tendance linéaire de
# chaque variable de densité, zone par zone, sur les 4 années connues (2017-2020),
# et on utilise cette valeur projetée (différente chaque année) comme entrée du RF.

ANNEES_CIBLE = list(range(2023, 2031))
ANNEES_CONNUES_DENSITE = sorted(density_stats['annee'].unique())  # ex: [2017, 2018, 2019, 2020]

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
    """Projette les variables de densité pour une année donnée, à partir de la
    tendance 2017-2020 de chaque zone. La projection brute est ensuite plafonnée
    à la fourchette [min, max] réellement observée pour CETTE zone sur 2017-2020 :
    on garde la direction de la tendance, sans extrapoler au-delà de ce que le
    modèle a vraiment vu, ce qui évite les valeurs délirantes sur les zones à
    faible historique projetées loin dans le temps (ex: 2030)."""
    X_projete = pd.DataFrame({'zone_id': zones['zone_id']})
    for feature in FEATURES:
        if feature == 'reference_historique':
            continue  # constante, pas une variable de densité à extrapoler
        valeurs_brutes = intercepts[feature].reindex(zones['zone_id']).values + \
                           pentes[feature].reindex(zones['zone_id']).values * annee_cible
        mini = bornes_min[feature].reindex(zones['zone_id']).values
        maxi = bornes_max[feature].reindex(zones['zone_id']).values
        valeurs_plafonnees = np.clip(valeurs_brutes, a_min=mini, a_max=maxi)
        X_projete[feature] = np.clip(valeurs_plafonnees, a_min=0, a_max=None)

    # Référence historique : constante par zone, ne se projette pas dans le temps
    X_projete = X_projete.merge(reference_zone, on='zone_id', how='left')
    return X_projete


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

transformer_vers_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
lon_deg, lat_deg = transformer_vers_4326.transform(zones['longitude'].values, zones['latitude'].values)
zones['lat_deg'] = lat_deg
zones['lon_deg'] = lon_deg

resultats_annuels = resultats_annuels.merge(zones[['zone_id', 'lat_deg', 'lon_deg']], on='zone_id', how='left')

TAILLE_GRILLE_DEG = 1.5

resultats_annuels['grille_lat'] = np.floor(resultats_annuels['lat_deg'] / TAILLE_GRILLE_DEG) * TAILLE_GRILLE_DEG
resultats_annuels['grille_lon'] = np.floor(resultats_annuels['lon_deg'] / TAILLE_GRILLE_DEG) * TAILLE_GRILLE_DEG

agrege = resultats_annuels.groupby(['grille_lat', 'grille_lon', 'annee']).agg(
    predit_total=('valeur_predite', 'sum'),
    reel_total=('valeur_reelle', lambda s: np.nan if s.isna().all() else s.sum()),
    reference_totale=('reference_historique', 'sum'),
    nb_zones=('zone_id', 'nunique'),
).reset_index()

COULEURS = {
    'Forte diminution': '#08306b',
    'Faible diminution': '#27ae60',
    'Stable': '#8e44ad',
    'Faible augmentation': '#f4d03f',
    'Forte augmentation': '#e74c3c',
    'Données insuffisantes': '#cccccc',
}


def categoriser(ref, valeur):
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


agrege['pct_evolution'] = np.where(
    agrege['reference_totale'] >= 0.5,
    (agrege['predit_total'] - agrege['reference_totale']) / agrege['reference_totale'] * 100,
    np.nan
)
agrege['categorie'] = agrege.apply(lambda r: categoriser(r['reference_totale'], r['predit_total']), axis=1)
agrege['couleur'] = agrege['categorie'].map(COULEURS)

agrege['predit_affiche'] = np.floor(agrege['predit_total']).astype(int)


print(f"\n{len(agrege)} lignes (cases x années, {agrege['annee'].nunique()} années : "
      f"{ANNEES_CIBLE[0]}-{ANNEES_CIBLE[-1]})")
print("\nRépartition par catégorie (toutes années confondues) :")
print(agrege['categorie'].value_counts())

evaluation = agrege[agrege['reel_total'].notna()].copy()
if len(evaluation) > 0:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(evaluation['reel_total'], evaluation['predit_total'])
    rmse = np.sqrt(mean_squared_error(evaluation['reel_total'], evaluation['predit_total']))
    erreur_moyenne_pct = ((evaluation['predit_total'] - evaluation['reel_total']).abs()
                           / evaluation['reel_total'].replace(0, np.nan)).mean() * 100

    for annee, groupe in evaluation.groupby('annee'):
        mae_annee = mean_absolute_error(groupe['reel_total'], groupe['predit_total'])
        print(f"  {int(annee)} : MAE = {mae_annee:.2f} accidents ({len(groupe)} cases)")
else:
    print("\nAucune année avec valeur réelle connue à évaluer.")

os.makedirs("data/processed", exist_ok=True)
agrege.to_csv("data/processed/predictions_random_forest_densite_2023_2030.csv", index=False)
