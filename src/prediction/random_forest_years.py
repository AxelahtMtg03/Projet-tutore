import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pyproj import Transformer
from scipy.spatial import cKDTree
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
os.makedirs(DATA_PATH, exist_ok=True)

TAILLE_ZONE_DEG = 1.5
DISTANCE_MAX_M = 50_000
CLASSES_ORDRE = ["Forte diminution", "Faible diminution", "Stable",
                 "Faible augmentation", "Forte augmentation", "Données insuffisantes"]
SEUIL_REFERENCE_MIN = 0.15
SEUIL_ECART_MIN = 0.5

DELTA_REPRESENTATIF = {
    "Forte diminution": -40,
    "Faible diminution": -15,
    "Stable": 0,
    "Faible augmentation": 15,
    "Forte augmentation": 40,
    "Données insuffisantes": 0,
}


def calculer_zone(lat, lon, taille=TAILLE_ZONE_DEG):
    zone_lat = np.floor(lat / taille) * taille
    zone_lon = np.floor(lon / taille) * taille
    return "z_" + zone_lat.round(1).astype(str) + "_" + zone_lon.round(1).astype(str)


def categoriser(ecart_absolu, pct, reference):
    if pd.isna(pct) or pd.isna(reference):
        return np.nan
    if reference < 0.5:
        return "Données insuffisantes"
    if abs(ecart_absolu) < 1.5:
        return "Stable"
    if pct < -30:
        return "Forte diminution"
    elif pct < -10:
        return "Faible diminution"
    elif pct < 10:
        return "Stable"
    elif pct < 30:
        return "Faible augmentation"
    else:
        return "Forte augmentation"


def run_classification_tendance_with_cv():
    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long", "annee"])
    accidents["annee"] = accidents["annee"].astype(int)
    density["annee"] = pd.to_datetime(density["time"]).dt.year

    accidents["zone"] = calculer_zone(accidents["lat"], accidents["long"])
    accidents_par_zone = accidents.groupby(["zone", "annee"]).size().reset_index(name="nb_accidents")
    zones = accidents_par_zone["zone"].unique()

    annees_historiques = list(range(2011, 2026))
    index_complet = pd.MultiIndex.from_product([zones, annees_historiques], names=["zone", "annee"])
    data = accidents_par_zone.set_index(["zone", "annee"]).reindex(index_complet, fill_value=0).reset_index()
    data["nb_accidents"] = data["nb_accidents"].astype(float)

    data["reference_5_ans"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).median()
    )
    data["accidents_suivants"] = data.groupby("zone")["nb_accidents"].shift(-1)
    data["reference_suivante"] = data.groupby("zone")["reference_5_ans"].shift(-1)
    data["ecart_absolu_suivant"] = data["accidents_suivants"] - data["reference_suivante"]
    data["evolution_suivante"] = np.where(
        data["reference_suivante"] > 0.5,
        (data["ecart_absolu_suivant"] / data["reference_suivante"]) * 100,
        np.nan
    )
    data["categorie"] = data.apply(
        lambda r: categoriser(r["ecart_absolu_suivant"], r["evolution_suivante"], r["reference_suivante"]),
        axis=1
    )

    for lag in [1, 2, 3, 4, 5]:
        data[f"nb_accidents_lag_{lag}"] = data.groupby("zone")["nb_accidents"].shift(lag)
    data["mediane_3"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).median()
    )
    data["mediane_5"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).median()
    )
    data["evolution_recente"] = np.where(
        data["mediane_5"] > 0, ((data["mediane_3"] - data["mediane_5"]) / data["mediane_5"]) * 100, 0
    )

    transformer_vers_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = transformer_vers_4326.transform(density["longitude"].values, density["latitude"].values)
    density["lat_deg"] = lat_deg
    density["lon_deg"] = lon_deg
    density["zone"] = calculer_zone(density["lat_deg"], density["lon_deg"])

    density_par_zone_annee = density.groupby(["zone", "annee"]).agg(
        densite_moyenne=("vd", "mean"),
        densite_max=("vd", "max"),
    ).reset_index()

    data = data.merge(density_par_zone_annee, on=["zone", "annee"], how="left")
    data[["densite_moyenne", "densite_max"]] = (
        data.groupby("zone")[["densite_moyenne", "densite_max"]].transform(lambda s: s.ffill().bfill())
    )
    moyenne_globale_densite = density_par_zone_annee[["densite_moyenne", "densite_max"]].mean()
    data["densite_moyenne"] = data["densite_moyenne"].fillna(moyenne_globale_densite["densite_moyenne"])
    data["densite_max"] = data["densite_max"].fillna(moyenne_globale_densite["densite_max"])

    fleet_total = fleet[fleet["type_navire"] == "Flotte totale"][["annee", "nombre_navires"]].drop_duplicates("annee")
    data = data.merge(fleet_total, on="annee", how="left")
    data["nombre_navires"] = data["nombre_navires"].ffill().bfill()

    FEATURES = ["annee", "densite_moyenne", "densite_max", "nombre_navires",
                "nb_accidents_lag_1", "nb_accidents_lag_2", "nb_accidents_lag_3",
                "nb_accidents_lag_4", "nb_accidents_lag_5",
                "mediane_3", "mediane_5", "evolution_recente"]

    data_modele = data.dropna(subset=FEATURES + ["categorie"]).copy()
    train = data_modele[data_modele["annee"] <= 2022].reset_index(drop=True)
    X_train, y_train = train[FEATURES], train["categorie"]

    annees_train_dispo = sorted(train["annee"].unique())
    tscv = TimeSeriesSplit(n_splits=5)
    splits = []
    for idx_train_annees, idx_test_annees in tscv.split(annees_train_dispo):
        annees_train_fold = [annees_train_dispo[j] for j in idx_train_annees]
        annees_test_fold = [annees_train_dispo[j] for j in idx_test_annees]
        idx_train = train.index[train["annee"].isin(annees_train_fold)].to_numpy()
        idx_test = train.index[train["annee"].isin(annees_test_fold)].to_numpy()
        if len(idx_train) > 0 and len(idx_test) > 0:
            splits.append((idx_train, idx_test))

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        param_grid, cv=splits, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    classes_modele = model.classes_

    resultats = []
    future_years = list(range(2023, 2031))
    historique = data[["zone", "annee", "nb_accidents", "densite_moyenne", "densite_max", "nombre_navires"]].copy()

    for zone in zones:
        zone_data = historique[historique["zone"] == zone].sort_values("annee").copy()
        accidents_list = zone_data["nb_accidents"].fillna(0).tolist()
        densite_list = zone_data["densite_moyenne"].fillna(0).tolist()
        densite_max_list = zone_data["densite_max"].fillna(0).tolist()
        navires_list = zone_data["nombre_navires"].fillna(0).tolist()

        for annee in future_years:
            if len(accidents_list) < 5:
                continue

            lag1, lag2, lag3, lag4, lag5 = (
                accidents_list[-1], accidents_list[-2], accidents_list[-3],
                accidents_list[-4], accidents_list[-5]
            )
            mediane_3 = np.median(accidents_list[-3:])
            mediane_5 = np.median(accidents_list[-5:])
            evolution_recente = ((mediane_3 - mediane_5) / mediane_5 * 100) if mediane_5 > 0 else 0

            X_future = pd.DataFrame(
                [[annee, densite_list[-1], densite_max_list[-1], navires_list[-1],
                  lag1, lag2, lag3, lag4, lag5,
                  mediane_3, mediane_5, evolution_recente]],
                columns=FEATURES
            )
            probabilites = model.predict_proba(X_future)[0]
            idx_predit = np.argmax(probabilites)
            categorie_predite = classes_modele[idx_predit]
            confiance = probabilites[idx_predit]

            resultats.append({
                "zone": zone,
                "annee": annee,
                "tendance": categorie_predite,
                "confiance": round(confiance * 100, 1),
            })

            if annee <= 2025:
                ligne_reelle = data[(data["zone"] == zone) & (data["annee"] == annee)]
                nb = float(ligne_reelle["nb_accidents"].iloc[0]) if len(ligne_reelle) > 0 else 0.0
            else:
                delta_pct_pondere = sum(
                    proba * DELTA_REPRESENTATIF[classe]
                    for classe, proba in zip(classes_modele, probabilites)
                )
                nb = max(0.0, mediane_5 * (1 + delta_pct_pondere / 100))

            accidents_list.append(nb)
            densite_list.append(densite_list[-1])
            densite_max_list.append(densite_max_list[-1])
            navires_list.append(navires_list[-1])

    df_futur = pd.DataFrame(resultats)
    output_file = DATA_PATH / "predictions_tendance_cv_2023_2030.csv"
    df_futur.to_csv(output_file, index=False)

    return df_futur, grid_search.best_params_


def test_accuracy_cv():
    try:
        df_pred = pd.read_csv(DATA_PATH / "predictions_tendance_cv_2023_2030.csv")
    except FileNotFoundError:
        print("Fichier de predictions non trouve")
        return

    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    accidents = accidents.dropna(subset=["lat", "long", "annee"])

    def assigner_zone(row):
        lat_zone = np.floor(row["lat"] / 1.5) * 1.5
        lon_zone = np.floor(row["long"] / 1.5) * 1.5
        return f"z_{lat_zone:.1f}_{lon_zone:.1f}"

    accidents["zone"] = accidents.apply(assigner_zone, axis=1)
    accidents_par_zone = accidents.groupby(["zone", "annee"]).size().reset_index(name="nb_accidents")

    ref_zone = accidents_par_zone[accidents_par_zone["annee"] <= 2022].groupby("zone")["nb_accidents"].median().reset_index()
    ref_zone.columns = ["zone", "reference_historique"]

    data = accidents_par_zone.merge(ref_zone, on="zone", how="left")
    data["reference_historique"] = data["reference_historique"].fillna(0)

    def categoriser(row):
        if row["reference_historique"] < 0.5:
            return "Donnees insuffisantes"
        evolution = (row["nb_accidents"] - row["reference_historique"]) / row["reference_historique"] * 100
        ecart_absolu = row["nb_accidents"] - row["reference_historique"]
        if abs(ecart_absolu) < 1.5:
            return "Stable"
        elif evolution < -30:
            return "Forte diminution"
        elif evolution < -10:
            return "Faible diminution"
        elif evolution < 10:
            return "Stable"
        elif evolution < 30:
            return "Faible augmentation"
        else:
            return "Forte augmentation"

    data["tendance_reelle"] = data.apply(categoriser, axis=1)
    data = data[data["annee"].between(2023, 2025)]

    df_compare = df_pred.merge(data[["zone", "annee", "tendance_reelle"]], on=["zone", "annee"], how="inner")
    df_compare = df_compare[df_compare["tendance_reelle"] != "Donnees insuffisantes"]

    if len(df_compare) == 0:
        print("Aucune correspondance")
        return

    df_compare["correct"] = df_compare["tendance"] == df_compare["tendance_reelle"]
    accuracy = df_compare["correct"].mean() * 100

    print("\n" + "=" * 60)
    print("ACCURACY - MODELE ANNUEL (AVEC CROSS-VALIDATION)")
    print("=" * 60)
    print(f"\nAccuracy globale: {accuracy:.1f}%")
    print(f"Zones comparees: {len(df_compare)}")

    print("\nRESUME PAR ANNEE:")
    for annee in [2023, 2024, 2025]:
        df_annee = df_compare[df_compare["annee"] == annee]
        if len(df_annee) > 0:
            acc = df_annee["correct"].mean() * 100
            print(f"  {annee}: {len(df_annee)} zones, Accuracy = {acc:.1f}%")

    print("\nRESUME PAR TENDANCE:")
    for classe in ["Forte diminution", "Faible diminution", "Stable", "Faible augmentation", "Forte augmentation"]:
        df_classe = df_compare[df_compare["tendance_reelle"] == classe]
        if len(df_classe) > 0:
            acc = df_classe["correct"].mean() * 100
            print(f"  {classe}: {len(df_classe)} zones, Accuracy = {acc:.1f}%")

    return df_compare



df_futur, best_params = run_classification_tendance_with_cv()
print(f"\nMeilleurs parametres: {best_params}")
test_accuracy_cv()