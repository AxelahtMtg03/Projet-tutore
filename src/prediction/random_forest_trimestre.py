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


def run_classification_tendance_trimestrielle():
    print("=" * 70)
    print("RANDOM FOREST - PREDICTION TRIMESTRIELLE (AVEC CV)")
    print("=" * 70)

    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long"])

    accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"], errors="coerce").dt.month
    accidents = accidents.dropna(subset=["mois"])
    accidents["trimestre"] = np.ceil(accidents["mois"] / 3).astype(int)

    density["mois"] = pd.to_datetime(density["time"]).dt.month
    density["trimestre"] = np.ceil(density["mois"] / 3).astype(int)
    density["annee"] = pd.to_datetime(density["time"]).dt.year

    transformer_vers_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_acc, y_acc = transformer_vers_3035.transform(accidents["long"].values, accidents["lat"].values)
    accidents["x"] = x_acc
    accidents["y"] = y_acc

    zones_natives = density[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    zones_natives["zone_id"] = zones_natives.index
    zone_tree = cKDTree(zones_natives[["latitude", "longitude"]].values)

    distances, nearest_idx = zone_tree.query(accidents[["y", "x"]].values)
    accidents["zone_id"] = zones_natives.loc[nearest_idx, "zone_id"].values
    accidents["distance_zone"] = distances
    accidents = accidents[accidents["distance_zone"] <= DISTANCE_MAX_M].copy()

    transformer_vers_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = transformer_vers_4326.transform(zones_natives["longitude"].values, zones_natives["latitude"].values)
    zones_natives["lat_deg"] = lat_deg
    zones_natives["lon_deg"] = lon_deg
    zones_natives["zone"] = calculer_zone(zones_natives["lat_deg"], zones_natives["lon_deg"])

    accidents = accidents.merge(zones_natives[["zone_id", "zone"]], on="zone_id", how="left")

    accidents_par_zone_trimestre = accidents.groupby(["zone", "annee", "trimestre"]).size().reset_index(name="nb_accidents")
    zones = accidents_par_zone_trimestre["zone"].unique()

    index_complet = pd.MultiIndex.from_product(
        [zones, range(2011, 2026), [1, 2, 3, 4]],
        names=["zone", "annee", "trimestre"]
    )

    data = accidents_par_zone_trimestre.set_index(["zone", "annee", "trimestre"]).reindex(index_complet, fill_value=0).reset_index()
    data["nb_accidents"] = data["nb_accidents"].astype(float)

    data["reference_12_trimestres"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=6).median()
    )
    data["accidents_suivants"] = data.groupby("zone")["nb_accidents"].shift(-1)
    data["reference_suivante"] = data.groupby("zone")["reference_12_trimestres"].shift(-1)
    data["ecart_absolu_suivant"] = data["accidents_suivants"] - data["reference_suivante"]
    data["evolution_suivante"] = np.where(
        data["reference_suivante"] > SEUIL_REFERENCE_MIN,
        (data["ecart_absolu_suivant"] / data["reference_suivante"]) * 100,
        np.nan
    )
    data["categorie"] = data.apply(
        lambda r: categoriser(r["ecart_absolu_suivant"], r["evolution_suivante"], r["reference_suivante"]),
        axis=1
    )

    for lag in [1, 2, 3, 4, 8, 12, 16, 20]:
        data[f"nb_accidents_lag_{lag}"] = data.groupby("zone")["nb_accidents"].shift(lag)

    data["mediane_4"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).median()
    )
    data["mediane_8"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=4).median()
    )
    data["mediane_12"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=6).median()
    )
    data["mediane_20"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).median()
    )
    data["evolution_recente"] = np.where(
        data["mediane_20"] > 0, ((data["mediane_4"] - data["mediane_20"]) / data["mediane_20"]) * 100, 0
    )

    density = density.merge(zones_natives[["latitude", "longitude", "zone"]], on=["latitude", "longitude"], how="left")

    density_par_zone_trimestre = density.groupby(["zone", "annee", "trimestre"]).agg(
        densite_moyenne=("vd", "mean"),
        densite_max=("vd", "max"),
    ).reset_index()

    data = data.merge(density_par_zone_trimestre, on=["zone", "annee", "trimestre"], how="left")
    data[["densite_moyenne", "densite_max"]] = (
        data.groupby("zone")[["densite_moyenne", "densite_max"]].transform(lambda s: s.ffill().bfill())
    )

    fleet_total = fleet[fleet["type_navire"] == "Flotte totale"][["annee", "nombre_navires"]].drop_duplicates("annee")
    data = data.merge(fleet_total, on="annee", how="left")
    data["nombre_navires"] = data["nombre_navires"].ffill().bfill()

    FEATURES = ["annee", "trimestre", "densite_moyenne", "densite_max", "nombre_navires",
                "nb_accidents_lag_1", "nb_accidents_lag_2", "nb_accidents_lag_3", "nb_accidents_lag_4",
                "nb_accidents_lag_8", "nb_accidents_lag_12", "nb_accidents_lag_16", "nb_accidents_lag_20",
                "mediane_4", "mediane_8", "mediane_12", "mediane_20", "evolution_recente"]

    data_modele = data.dropna(subset=FEATURES + ["categorie"]).copy()
    train = data_modele[data_modele["annee"] < 2023].reset_index(drop=True)
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
    future_trimestres = [(annee, trim) for annee in range(2023, 2031) for trim in [1, 2, 3, 4]]

    historique = data[["zone", "annee", "trimestre", "nb_accidents", "densite_moyenne", "densite_max", "nombre_navires"]].copy()

    etat_zones = {}
    for zone in zones:
        zone_data = historique[historique["zone"] == zone].sort_values(["annee", "trimestre"])
        etat_zones[zone] = {
            "accidents": zone_data["nb_accidents"].fillna(0).tolist(),
            "densite": zone_data["densite_moyenne"].fillna(0).tolist(),
            "densite_max": zone_data["densite_max"].fillna(0).tolist(),
            "navires": zone_data["nombre_navires"].fillna(0).tolist(),
        }

    zones_valides = [z for z in zones if len(etat_zones[z]["accidents"]) >= 20]
    print(f"{len(zones_valides)}/{len(zones)} zones avec assez d'historique")

    for annee, trimestre in future_trimestres:
        lignes_features = []
        for zone in zones_valides:
            s = etat_zones[zone]
            accidents_list = s["accidents"]
            lag1, lag2, lag3, lag4 = accidents_list[-1], accidents_list[-2], accidents_list[-3], accidents_list[-4]
            lag8, lag12, lag16, lag20 = accidents_list[-8], accidents_list[-12], accidents_list[-16], accidents_list[-20]
            mediane_4 = np.median(accidents_list[-4:])
            mediane_8 = np.median(accidents_list[-8:])
            mediane_12 = np.median(accidents_list[-12:])
            mediane_20 = np.median(accidents_list[-20:])
            evolution_recente = ((mediane_4 - mediane_20) / mediane_20 * 100) if mediane_20 > 0 else 0

            lignes_features.append([
                annee, trimestre, s["densite"][-1], s["densite_max"][-1], s["navires"][-1],
                lag1, lag2, lag3, lag4, lag8, lag12, lag16, lag20,
                mediane_4, mediane_8, mediane_12, mediane_20, evolution_recente
            ])

        X_future = pd.DataFrame(lignes_features, columns=FEATURES)
        probabilites_toutes_zones = model.predict_proba(X_future)

        for i, zone in enumerate(zones_valides):
            probabilites = probabilites_toutes_zones[i]
            idx_predit = np.argmax(probabilites)
            categorie_predite = classes_modele[idx_predit]
            confiance = probabilites[idx_predit]

            resultats.append({
                "zone": zone,
                "annee": annee,
                "trimestre": trimestre,
                "trimestre_annee": f"{annee}-T{trimestre}",
                "tendance": categorie_predite,
                "confiance": round(confiance * 100, 1),
            })

            s = etat_zones[zone]
            ligne_reelle = data[(data["zone"] == zone) & (data["annee"] == annee) & (data["trimestre"] == trimestre)]
            if len(ligne_reelle) > 0:
                nb = float(ligne_reelle["nb_accidents"].iloc[0])
            else:
                mediane_20_actuelle = np.median(s["accidents"][-20:])
                delta_pct_pondere = sum(
                    proba * DELTA_REPRESENTATIF[classe]
                    for classe, proba in zip(classes_modele, probabilites)
                )
                nb = max(0.0, mediane_20_actuelle * (1 + delta_pct_pondere / 100))

            s["accidents"].append(nb)
            s["densite"].append(s["densite"][-1])
            s["densite_max"].append(s["densite_max"][-1])
            s["navires"].append(s["navires"][-1])

    df_futur = pd.DataFrame(resultats)
    output_file = DATA_PATH / "predictions_tendance_trimestrielle_2023_2030.csv"
    df_futur.to_csv(output_file, index=False)
    print(f"\nCSV genere: {output_file} ({len(df_futur)} predictions)")

    return df_futur, grid_search.best_params_


def test_accuracy_trimestriel():
    try:
        df_pred = pd.read_csv(DATA_PATH / "predictions_tendance_trimestrielle_2023_2030.csv")
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
    accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"]).dt.month
    accidents["trimestre"] = np.ceil(accidents["mois"] / 3).astype(int)

    accidents_par_zone = accidents.groupby(["zone", "annee", "trimestre"]).size().reset_index(name="nb_accidents")

    ref_zone = accidents_par_zone[accidents_par_zone["annee"] <= 2022].groupby(["zone", "trimestre"])["nb_accidents"].median().reset_index()
    ref_zone.columns = ["zone", "trimestre", "reference_historique"]

    data = accidents_par_zone.merge(ref_zone, on=["zone", "trimestre"], how="left")
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

    df_pred["trimestre"] = df_pred["trimestre"].astype(int)
    data["trimestre"] = data["trimestre"].astype(int)

    df_compare = df_pred.merge(data[["zone", "annee", "trimestre", "tendance_reelle"]], on=["zone", "annee", "trimestre"], how="inner")
    df_compare = df_compare[df_compare["tendance_reelle"] != "Donnees insuffisantes"]

    if len(df_compare) == 0:
        print("Aucune correspondance")
        return

    df_compare["correct"] = df_compare["tendance"] == df_compare["tendance_reelle"]
    accuracy = df_compare["correct"].mean() * 100

    print("\n" + "=" * 60)
    print("ACCURACY - MODELE TRIMESTRIEL (AVEC CV)")
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


df_futur, best_params = run_classification_tendance_trimestrielle()
print(f"\nMeilleurs parametres: {best_params}")
test_accuracy_trimestriel()