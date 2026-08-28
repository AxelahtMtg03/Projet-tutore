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


# ============================================================
# FONCTION 2 : MODELE CLASSIFICATION DES TENDANCES
# ============================================================

def run_classification_tendance():
    """
    Random Forest - prédiction de la tendance des accidents.

    Le modèle apprend à prédire la tendance d'une année
    à partir uniquement des informations disponibles avant
    cette année.

    Sortie :
        - Forte diminution
        - Faible diminution
        - Stable
        - Faible augmentation
        - Forte augmentation
    """

    print("=" * 70)
    print("RANDOM FOREST - PREDICTION DES TENDANCES")
    print("=" * 70)

    # ============================================================
    # 1. CHARGEMENT DES DONNEES
    # ============================================================

    print("\n1. Chargement des donnees...")

    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long", "annee"])
    accidents["annee"] = accidents["annee"].astype(int)

    print(f" Accidents : {len(accidents)}")
    print(f" Flotte : {len(fleet)}")
    print(f" Densite : {len(density)}")


    # ============================================================
    # 2. CREATION DE LA GRILLE 1.5° x 1.5°
    # ============================================================

    print("\n2. Creation des zones...")

    def assigner_zone(row):
        lat_zone = np.floor(row["lat"] / 1.5) * 1.5
        lon_zone = np.floor(row["long"] / 1.5) * 1.5

        return f"z_{lat_zone:.1f}_{lon_zone:.1f}"

    accidents["zone"] = accidents.apply(assigner_zone, axis=1)

    accidents_par_zone = (
        accidents
        .groupby(["zone", "annee"])
        .size()
        .reset_index(name="nb_accidents")
    )

    print(
        f" {accidents_par_zone['zone'].nunique()} zones identifiees"
    )


    # ============================================================
    # 3. COMPLETER LES ANNEES MANQUANTES
    # ============================================================

    # Important :
    # Si une zone n'a aucun accident une année donnée,
    # elle doit avoir 0 accident et non pas une ligne absente.

    annees_historiques = list(range(2011, 2026))
    zones = accidents_par_zone["zone"].unique()

    index_complet = pd.MultiIndex.from_product(
        [zones, annees_historiques],
        names=["zone", "annee"]
    )

    data = (
        accidents_par_zone
        .set_index(["zone", "annee"])
        .reindex(index_complet, fill_value=0)
        .reset_index()
    )

    data["nb_accidents"] = data["nb_accidents"].astype(float)


    # ============================================================
    # 4. REFERENCE HISTORIQUE
    # ============================================================

    # Pour chaque année, la référence correspond à la moyenne
    # des années précédentes.
    #
    # Exemple :
    # Pour prédire 2023 -> moyenne 2018-2022
    # Pour prédire 2024 -> moyenne 2019-2023
    # etc.

    data["reference_5_ans"] = (
        data
        .groupby("zone")["nb_accidents"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    )


    # ============================================================
    # 5. CREATION DE LA CIBLE : TENDANCE DE L'ANNEE SUIVANTE
    # ============================================================

    # On veut prédire la tendance de l'année suivante.
    #
    # Exemple :
    # Les informations 2022 servent à prédire la tendance 2023.

    data["accidents_suivants"] = (
        data.groupby("zone")["nb_accidents"].shift(-1)
    )

    data["reference_suivante"] = (
        data.groupby("zone")["reference_5_ans"].shift(-1)
    )

    data["evolution_suivante"] = np.where(
        data["reference_suivante"] > 0.5,
        (
            (data["accidents_suivants"] -
             data["reference_suivante"])
            / data["reference_suivante"]
        ) * 100,
        np.nan
    )


    def categoriser(evolution):

        if pd.isna(evolution):
            return np.nan

        if evolution < -30:
            return "Forte diminution"

        elif evolution < -10:
            return "Faible diminution"

        elif evolution < 10:
            return "Stable"

        elif evolution < 30:
            return "Faible augmentation"

        else:
            return "Forte augmentation"


    data["categorie"] = data["evolution_suivante"].apply(
        categoriser
    )


    # ============================================================
    # 6. VARIABLES TEMPORELLES
    # ============================================================

    print("\n3. Creation des variables...")

    # Lags = uniquement informations du passé

    for lag in [1, 2, 3]:
        data[f"nb_accidents_lag_{lag}"] = (
            data.groupby("zone")["nb_accidents"]
            .shift(lag)
        )


    # Moyenne des 3 dernières années AVANT l'année courante

    data["moyenne_3"] = (
        data.groupby("zone")["nb_accidents"]
        .transform(
            lambda x: x.shift(1).rolling(
                3,
                min_periods=2
            ).mean()
        )
    )


    # Moyenne des 5 dernières années AVANT l'année courante

    data["moyenne_5"] = (
        data.groupby("zone")["nb_accidents"]
        .transform(
            lambda x: x.shift(1).rolling(
                5,
                min_periods=3
            ).mean()
        )
    )


    # Evolution récente :
    # comparaison de la moyenne 3 ans avec moyenne 5 ans

    data["evolution_recente"] = np.where(
        data["moyenne_5"] > 0,
        (
            (data["moyenne_3"] - data["moyenne_5"])
            / data["moyenne_5"]
        ) * 100,
        0
    )


    # ============================================================
    # 7. DENSITE MARITIME
    # ============================================================

    density["annee"] = pd.to_datetime(
        density["time"]
    ).dt.year

    density_par_annee = (
        density
        .groupby("annee")
        .agg(
            densite_moyenne=("vd", "mean"),
            densite_max=("vd", "max")
        )
        .reset_index()
    )

    data = data.merge(
        density_par_annee,
        on="annee",
        how="left"
    )


    # Pour les années sans densité :
    # on utilise la dernière valeur connue.
    #
    # La densité disponible s'arrêtant avant les années futures,
    # cette hypothèse doit être mentionnée dans le rapport.

    data["densite_moyenne"] = (
        data["densite_moyenne"]
        .ffill()
        .bfill()
    )

    data["densite_max"] = (
        data["densite_max"]
        .ffill()
        .bfill()
    )


    # ============================================================
    # 8. FLOTTES
    # ============================================================

    fleet_total = fleet[
        fleet["type_navire"] == "Flotte totale"
    ][
        ["annee", "nombre_navires"]
    ].drop_duplicates("annee")

    data = data.merge(
        fleet_total,
        on="annee",
        how="left"
    )

    data["nombre_navires"] = (
        data["nombre_navires"]
        .ffill()
        .bfill()
    )


    # ============================================================
    # 9. VARIABLES FINALES
    # ============================================================

    FEATURES = [
        "annee",
        "densite_moyenne",
        "densite_max",
        "nombre_navires",

        "nb_accidents_lag_1",
        "nb_accidents_lag_2",
        "nb_accidents_lag_3",

        "moyenne_3",
        "moyenne_5",

        "evolution_recente"
    ]

    classes_ordre = [
        "Forte diminution",
        "Faible diminution",
        "Stable",
        "Faible augmentation",
        "Forte augmentation"
    ]

    data["categorie_num"] = (
        data["categorie"]
        .map({
            classe: i
            for i, classe in enumerate(classes_ordre)
        })
    )


    # ============================================================
    # 10. SUPPRESSION DES LIGNES INCOMPLETES
    # ============================================================

    data_modele = data.dropna(
        subset=FEATURES + ["categorie_num"]
    ).copy()

    print(
        f" Lignes utilisables : {len(data_modele)}"
    )


    # ============================================================
    # 11. TRAIN / TEST TEMPOREL
    # ============================================================

    # 2011-2022 = apprentissage
    # 2023-2025 = test

    train = data_modele[
        data_modele["annee"] <= 2022
    ]

    test = data_modele[
        data_modele["annee"].between(2023, 2025)
    ]

    X_train = train[FEATURES]
    y_train = train["categorie_num"].astype(int)

    X_test = test[FEATURES]
    y_test = test["categorie_num"].astype(int)

    print("\n4. Separation temporelle")
    print(f" Entrainement : {len(train)} lignes")
    print(f" Test : {len(test)} lignes")


    # ============================================================
    # 12. RANDOM FOREST
    # ============================================================

    print("\n5. Entrainement du Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )


    # ============================================================
    # 13. EVALUATION 2023-2025
    # ============================================================

    print("\n")
    print("=" * 70)
    print("EVALUATION 2023-2025")
    print("=" * 70)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nAccuracy : {accuracy:.3f}"
        f" ({accuracy * 100:.1f} %)"
    )

    print("\nRapport de classification :")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(classes_ordre))),
            target_names=classes_ordre,
            zero_division=0
        )
    )


    # ============================================================
    # 14. MATRICE DE CONFUSION
    # ============================================================

    matrice = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(classes_ordre)))
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matrice,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes_ordre,
        yticklabels=classes_ordre
    )

    plt.xlabel("Tendance prédite")
    plt.ylabel("Tendance réelle")

    plt.title(
        "Matrice de confusion - "
        "Prédiction de tendance"
    )

    plt.tight_layout()

    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH /
        "confusion_matrix_tendance.png",
        dpi=300
    )

    plt.close()


    # ============================================================
    # 15. IMPORTANCE DES VARIABLES
    # ============================================================

    importance = pd.DataFrame({
        "Parametre": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nImportance des variables :")

    for _, row in importance.iterrows():

        print(
            f" {row['Parametre']:25s} "
            f"{row['Importance']:.3f}"
        )


    # ============================================================
    # 16. PREDICTION 2023-2030
    # ============================================================

    print("\n")
    print("=" * 70)
    print("PREDICTION DES TENDANCES 2023-2030")
    print("=" * 70)

    # On travaille zone par zone.

    resultats = []

    future_years = list(
        range(2023, 2031)
    )

    # Copie des données historiques
    historique = data[
        [
            "zone",
            "annee",
            "nb_accidents",
            "densite_moyenne",
            "densite_max",
            "nombre_navires"
        ]
    ].copy()


    # ============================================================
    # Fonction pour construire les variables
    # d'une année donnée
    # ============================================================

    def construire_features(
        historique_zone,
        annee
    ):

        historique_zone = (
            historique_zone
            .sort_values("annee")
            .copy()
        )

        accidents = historique_zone[
            "nb_accidents"
        ].tolist()

        if len(accidents) < 5:
            return None

        lag1 = accidents[-1]
        lag2 = accidents[-2]
        lag3 = accidents[-3]

        moyenne_3 = np.mean(
            accidents[-3:]
        )

        moyenne_5 = np.mean(
            accidents[-5:]
        )

        if moyenne_5 > 0:
            evolution_recente = (
                (moyenne_3 - moyenne_5)
                / moyenne_5
            ) * 100
        else:
            evolution_recente = 0

        derniere_densite = (
            historique_zone["densite_moyenne"]
            .dropna()
            .iloc[-1]
        )

        derniere_densite_max = (
            historique_zone["densite_max"]
            .dropna()
            .iloc[-1]
        )

        dernier_nombre_navires = (
            historique_zone["nombre_navires"]
            .dropna()
            .iloc[-1]
        )

        return pd.DataFrame([{
            "annee": annee,
            "densite_moyenne": derniere_densite,
            "densite_max": derniere_densite_max,
            "nombre_navires": dernier_nombre_navires,

            "nb_accidents_lag_1": lag1,
            "nb_accidents_lag_2": lag2,
            "nb_accidents_lag_3": lag3,

            "moyenne_3": moyenne_3,
            "moyenne_5": moyenne_5,

            "evolution_recente":
                evolution_recente
        }])


    # ============================================================
    # PREDICTIONS
    # ============================================================

    for zone in zones:

        zone_data = historique[
            historique["zone"] == zone
        ].sort_values("annee").copy()

        # On commence avec les données historiques.
        donnees_zone = zone_data.copy()

        for annee in future_years:

            X_future = construire_features(
                donnees_zone,
                annee
            )

            if X_future is None:
                continue

            prediction = model.predict(
                X_future[FEATURES]
            )[0]

            probabilites = model.predict_proba(
                X_future[FEATURES]
            )[0]

            tendance = classes_ordre[
                int(prediction)
            ]

            confiance = (
                probabilites[int(prediction)]
                * 100
            )

            resultats.append({
                "zone": zone,
                "annee": annee,
                "tendance": tendance,
                "classe": int(prediction),
                "confiance": round(
                    confiance,
                    1
                )
            })


            # ----------------------------------------------------
            # Pour les années réelles :
            # on utilise les vrais accidents.
            # ----------------------------------------------------

            if annee <= 2025:

                valeur_reelle = data[
                    (data["zone"] == zone)
                    &
                    (data["annee"] == annee)
                ]["nb_accidents"]

                if len(valeur_reelle) > 0:

                    nb = float(
                        valeur_reelle.iloc[0]
                    )

                else:
                    nb = 0


            # ----------------------------------------------------
            # Pour 2026-2030 :
            #
            # Nous n'avons pas les accidents futurs.
            #
            # On conserve donc le dernier niveau connu
            # comme hypothèse de trafic/accidents.
            #
            # IMPORTANT :
            # La carte affiche uniquement la tendance.
            # ----------------------------------------------------

            else:

                nb = float(
                    donnees_zone
                    ["nb_accidents"]
                    .iloc[-1]
                )


            # Ajouter l'année à l'historique
            # pour permettre aux variables temporelles
            # d'évoluer.

            nouvelle_ligne = pd.DataFrame([{
                "zone": zone,
                "annee": annee,
                "nb_accidents": nb,
                "densite_moyenne":
                    donnees_zone[
                        "densite_moyenne"
                    ].iloc[-1],
                "densite_max":
                    donnees_zone[
                        "densite_max"
                    ].iloc[-1],
                "nombre_navires":
                    donnees_zone[
                        "nombre_navires"
                    ].iloc[-1]
            }])

            donnees_zone = pd.concat(
                [
                    donnees_zone,
                    nouvelle_ligne
                ],
                ignore_index=True
            )


    df_futur = pd.DataFrame(
        resultats
    )


    # ============================================================
    # 17. AFFICHAGE DU RESULTAT
    # ============================================================

    print("\nDistribution des tendances :")

    for annee in future_years:

        print(f"\n {annee} :")

        sous_df = df_futur[
            df_futur["annee"] == annee
        ]

        for classe in classes_ordre:

            nombre = len(
                sous_df[
                    sous_df["tendance"] == classe
                ]
            )

            print(
                f" {classe:25s} : "
                f"{nombre} zones"
            )


    # ============================================================
    # 18. SAUVEGARDE
    # ============================================================

    output_file = (
        DATA_PATH /
        "predictions_tendance_2023_2030.csv"
    )

    df_futur.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nPredictions sauvegardees : "
        f"{output_file}"
    )

    print("\nTermine.")

    return df_futur



# 1. Modele regression (original)
# print("1. MODELE REGRESSION (original)")
# print("-"*40)
# agrege = run_regression_prediction()

# 2. Modele classification (tendance)
print("2. MODELE CLASSIFICATION (tendance)")
print("-"*40)
df_tendance = run_classification_tendance()