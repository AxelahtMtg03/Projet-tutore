import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pyproj import Transformer
from scipy.spatial import cKDTree
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
os.makedirs(DATA_PATH, exist_ok=True)

TAILLE_ZONE_DEG = 1.5
DISTANCE_MAX_M = 50_000
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


def categoriser_avec_sous_classes(ecart_absolu, pct, reference):
    if pd.isna(pct) or pd.isna(reference):
        return np.nan
    if reference < 0.5:
        return "Données insuffisantes"

    if abs(ecart_absolu) < 0.5:
        return "Stable_c1"
    elif abs(ecart_absolu) < 1.0:
        return "Stable_c2"
    elif abs(ecart_absolu) < 1.5:
        return "Stable_c3"

    if pct < -30:
        return "Forte diminution"
    elif pct < -10:
        return "Faible diminution"
    elif pct < 10:
        return "Stable_c3"
    elif pct < 30:
        return "Faible augmentation"
    else:
        return "Forte augmentation"


def run_classification_tendance_mensuelle_subdivision():
    print("=" * 70)
    print("RANDOM FOREST - PREDICTION MENSUELLE AVEC SUBDIVISION")
    print("=" * 70)

    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long"])
    accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"], errors="coerce").values.astype("datetime64[M]")
    accidents = accidents.dropna(subset=["mois"])

    density["mois"] = pd.to_datetime(density["time"]).values.astype("datetime64[M]")

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

    accidents_par_zone_mois = accidents.groupby(["zone", "mois"]).size().reset_index(name="nb_accidents")
    zones = accidents_par_zone_mois["zone"].unique()

    mois_min = accidents_par_zone_mois["mois"].min()
    mois_max = accidents_par_zone_mois["mois"].max()
    mois_historiques = pd.date_range(mois_min, mois_max, freq="MS")
    print(f"Historique : {mois_min.date()} -> {mois_max.date()} ({len(mois_historiques)} mois)")

    index_complet = pd.MultiIndex.from_product([zones, mois_historiques], names=["zone", "mois"])
    data = accidents_par_zone_mois.set_index(["zone", "mois"]).reindex(index_complet, fill_value=0).reset_index()
    data["nb_accidents"] = data["nb_accidents"].astype(float)
    data["annee"] = data["mois"].dt.year
    data["mois_num"] = data["mois"].dt.month

    data["mois_sin"] = np.sin(2 * np.pi * data["mois_num"] / 12)
    data["mois_cos"] = np.cos(2 * np.pi * data["mois_num"] / 12)

    data["reference_12_mois"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=6).median()
    )
    data["accidents_suivant"] = data.groupby("zone")["nb_accidents"].shift(-1)
    data["reference_suivante"] = data.groupby("zone")["reference_12_mois"].shift(-1)
    data["ecart_absolu_suivant"] = data["accidents_suivant"] - data["reference_suivante"]
    data["evolution_suivante"] = np.where(
        data["reference_suivante"] > SEUIL_REFERENCE_MIN,
        (data["ecart_absolu_suivant"] / data["reference_suivante"]) * 100,
        np.nan
    )

    data["categorie"] = data.apply(
        lambda r: categoriser_avec_sous_classes(r["ecart_absolu_suivant"], r["evolution_suivante"], r["reference_suivante"]),
        axis=1
    )

    print("\nDistribution des categories avec subdivision :")
    print(data["categorie"].value_counts(dropna=False))

    for lag in [1, 2, 3, 6, 12, 24]:
        data[f"nb_accidents_lag_{lag}"] = data.groupby("zone")["nb_accidents"].shift(lag)

    data["mediane_3"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).median()
    )
    data["mediane_6"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=3).median()
    )
    data["mediane_12"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=6).median()
    )
    data["mediane_24"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(24, min_periods=12).median()
    )
    data["evolution_recente"] = np.where(
        data["mediane_24"] > 0, ((data["mediane_3"] - data["mediane_24"]) / data["mediane_24"]) * 100, 0
    )

    density = density.merge(zones_natives[["latitude", "longitude", "zone"]], on=["latitude", "longitude"], how="left")

    density_par_zone_mois = density.groupby(["zone", "mois"]).agg(
        densite_moyenne=("vd", "mean"),
        densite_max=("vd", "max"),
    ).reset_index()

    data = data.merge(density_par_zone_mois, on=["zone", "mois"], how="left")

    data[["densite_moyenne", "densite_max"]] = (
        data.groupby("zone")[["densite_moyenne", "densite_max"]].transform(lambda s: s.ffill().bfill())
    )
    moyenne_globale_densite = density_par_zone_mois[["densite_moyenne", "densite_max"]].mean()
    data["densite_moyenne"] = data["densite_moyenne"].fillna(moyenne_globale_densite["densite_moyenne"])
    data["densite_max"] = data["densite_max"].fillna(moyenne_globale_densite["densite_max"])

    fleet_total = fleet[fleet["type_navire"] == "Flotte totale"][["annee", "nombre_navires"]].drop_duplicates("annee")
    data = data.merge(fleet_total, on="annee", how="left")
    data["nombre_navires"] = data["nombre_navires"].ffill().bfill()

    FEATURES = ["annee", "mois_sin", "mois_cos", "densite_moyenne", "densite_max", "nombre_navires",
                "nb_accidents_lag_1", "nb_accidents_lag_2", "nb_accidents_lag_3",
                "nb_accidents_lag_6", "nb_accidents_lag_12", "nb_accidents_lag_24",
                "mediane_3", "mediane_6", "mediane_12", "mediane_24", "evolution_recente"]

    data_modele = data.dropna(subset=FEATURES + ["categorie"]).copy()
    date_coupure = pd.Timestamp("2023-01-01")
    train = data_modele[data_modele["mois"] < date_coupure].reset_index(drop=True)

    # ============================================================
    # CREATION DES CIBLES
    # ============================================================
    # Cible 1 : Stable vs Non-Stable
    train["cible_stable"] = train["categorie"].str.startswith("Stable").astype(int)

    # Cible 2 : Classe précise (pour tous)
    train["cible_classe"] = train["categorie"].apply(
        lambda x: "Stable" if x.startswith("Stable") else x
    )

    print("\nDistribution des cibles :")
    print(f"  Stable : {train['cible_stable'].sum()} ({train['cible_stable'].mean()*100:.1f}%)")
    print(f"  Non-Stable : {(1-train['cible_stable']).sum()} ({(1-train['cible_stable']).mean()*100:.1f}%)")
    print(f"\n  Classes précises :")
    print(train["cible_classe"].value_counts())

    # ============================================================
    # ENTRAINEMENT DES MODELES
    # ============================================================
    print("\nEntrainement des modeles...")

    # Modèle 1 : Stable vs Non-Stable
    model_stable = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model_stable.fit(train[FEATURES], train["cible_stable"])

    # Modèle 2 : Classe précise
    model_classe = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model_classe.fit(train[FEATURES], train["cible_classe"])

    print("2 modeles entraines")

    # ============================================================
    # PREDICTIONS
    # ============================================================
    mois_futurs = pd.date_range("2023-01-01", "2030-12-01", freq="MS")
    dernier_mois_connu = mois_max

    historique = data[["zone", "mois", "nb_accidents", "densite_moyenne", "densite_max", "nombre_navires"]].copy()

    etat_zones = {}
    for zone in zones:
        zone_data = historique[historique["zone"] == zone].sort_values("mois")
        etat_zones[zone] = {
            "accidents": zone_data["nb_accidents"].fillna(0).tolist(),
            "densite": zone_data["densite_moyenne"].fillna(0).tolist(),
            "densite_max": zone_data["densite_max"].fillna(0).tolist(),
            "navires": zone_data["nombre_navires"].fillna(0).tolist(),
        }

    data_recent = data[data["annee"].between(2023, 2025)]
    zones_avec_accidents = data_recent[data_recent["nb_accidents"] > 0]["zone"].unique()

    zones_valides = [z for z in zones if z in zones_avec_accidents and len(etat_zones[z]["accidents"]) >= 24]
    print(f"{len(zones_valides)}/{len(zones)} zones avec assez d'historique et accidents")

    resultats = []

    for mois in mois_futurs:
        annee_courante = mois.year
        mois_num = mois.month
        mois_sin = np.sin(2 * np.pi * mois_num / 12)
        mois_cos = np.cos(2 * np.pi * mois_num / 12)

        lignes_features = []
        for zone in zones_valides:
            s = etat_zones[zone]
            accidents_list = s["accidents"]
            lag1, lag2, lag3 = accidents_list[-1], accidents_list[-2], accidents_list[-3]
            lag6, lag12, lag24 = accidents_list[-6], accidents_list[-12], accidents_list[-24]
            mediane_3 = np.median(accidents_list[-3:])
            mediane_6 = np.median(accidents_list[-6:])
            mediane_12 = np.median(accidents_list[-12:])
            mediane_24 = np.median(accidents_list[-24:])
            evolution_recente = ((mediane_3 - mediane_24) / mediane_24 * 100) if mediane_24 > 0 else 0

            lignes_features.append([
                annee_courante, mois_sin, mois_cos, s["densite"][-1], s["densite_max"][-1], s["navires"][-1],
                lag1, lag2, lag3, lag6, lag12, lag24,
                mediane_3, mediane_6, mediane_12, mediane_24, evolution_recente
            ])

        X_future = pd.DataFrame(lignes_features, columns=FEATURES)

        proba_stable = model_stable.predict_proba(X_future)
        proba_classe = model_classe.predict_proba(X_future)
        classes_classe = model_classe.classes_

        for i, zone in enumerate(zones_valides):
            if proba_stable[i, 1] > 0.5:
                categorie_predite = "Stable"
                confiance = proba_stable[i, 1]
            else:
                idx_classe = np.argmax(proba_classe[i])
                categorie_predite = classes_classe[idx_classe]
                confiance = proba_classe[i, idx_classe]

            resultats.append({
                "zone": zone,
                "mois": mois.strftime("%Y-%m"),
                "annee": annee_courante,
                "mois_num": mois_num,
                "tendance": categorie_predite,
                "confiance": round(confiance * 100, 1),
            })

            s = etat_zones[zone]
            if mois <= dernier_mois_connu:
                ligne_reelle = data[(data["zone"] == zone) & (data["mois"] == mois)]
                nb = float(ligne_reelle["nb_accidents"].iloc[0]) if len(ligne_reelle) > 0 else 0.0
            else:
                mediane_24_actuelle = np.median(s["accidents"][-24:])
                nb = max(0.0, mediane_24_actuelle)

            s["accidents"].append(nb)
            s["densite"].append(s["densite"][-1])
            s["densite_max"].append(s["densite_max"][-1])
            s["navires"].append(s["navires"][-1])

    df_futur = pd.DataFrame(resultats)
    output_file = DATA_PATH / "predictions_tendance_mensuelle_subdivision.csv"
    df_futur.to_csv(output_file, index=False)
    print(f"\nCSV genere: {output_file} ({len(df_futur)} predictions)")

    return df_futur


def test_accuracy_subdivision():
    try:
        df_pred = pd.read_csv(DATA_PATH / "predictions_tendance_mensuelle_subdivision.csv")
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
    accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"]).dt.month.astype(int)

    accidents_par_zone = accidents.groupby(["zone", "annee", "mois"]).size().reset_index(name="nb_accidents")

    ref_zone = accidents_par_zone[accidents_par_zone["annee"] <= 2022].groupby(["zone", "mois"])["nb_accidents"].median().reset_index()
    ref_zone.columns = ["zone", "mois", "reference_historique"]

    data = accidents_par_zone.merge(ref_zone, on=["zone", "mois"], how="left")
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

    df_pred_eval = df_pred.copy()
    if df_pred_eval["mois"].dtype == "object":
        df_pred_eval["mois"] = df_pred_eval["mois"].str.split('-').str[1].astype(int)
    else:
        df_pred_eval["mois"] = df_pred_eval["mois"].astype(int)

    df_compare = df_pred_eval.merge(data[["zone", "annee", "mois", "tendance_reelle"]], on=["zone", "annee", "mois"], how="inner")
    df_compare = df_compare[df_compare["tendance_reelle"] != "Donnees insuffisantes"]

    if len(df_compare) == 0:
        print("Aucune correspondance")
        return

    df_compare["correct"] = df_compare["tendance"] == df_compare["tendance_reelle"]
    accuracy = df_compare["correct"].mean() * 100

    print("\n" + "=" * 60)
    print("ACCURACY - MODELE MENSUEL AVEC SUBDIVISION")
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


if __name__ == "__main__":
    df_futur = run_classification_tendance_mensuelle_subdivision()
    test_accuracy_subdivision()