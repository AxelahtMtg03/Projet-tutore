import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pyproj import Transformer
from pathlib import Path

# ============================================================
# CHEMINS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
os.makedirs(DATA_PATH, exist_ok=True)

TAILLE_ZONE_DEG = 1.5

CLASSES_ORDRE = ["Forte diminution", "Faible diminution", "Stable",
                 "Faible augmentation", "Forte augmentation", "Données insuffisantes"]

# Delta représentatif (%) associé à chaque catégorie, utilisé pour convertir une
# prédiction catégorielle en une valeur numérique lors de la boucle autorégressive
# (nécessaire pour recalculer les lags/moyennes des années suivantes).

DELTA_REPRESENTATIF = {
    "Forte diminution": -40,
    "Faible diminution": -15,
    "Stable": 0,
    "Faible augmentation": 15,
    "Forte augmentation": 40,
    "Données insuffisantes": 0,
}

def calculer_zone(lat, lon, taille=TAILLE_ZONE_DEG):
    """Assigne un identifiant de zone (grille en degrés) à des séries lat/lon."""
    zone_lat = np.floor(lat / taille) * taille
    zone_lon = np.floor(lon / taille) * taille
    return "z_" + zone_lat.round(1).astype(str) + "_" + zone_lon.round(1).astype(str)


def categoriser(ecart_absolu, pct, reference):
    """Catégorise une évolution : garde-fou sur la référence ET sur l'écart absolu,
    pour éviter les faux pourcentages énormes sur les zones à faible historique
    (ex: 0 accident au lieu de 1.1 en moyenne -> -100% qui n'a pas de sens en soi)."""
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


# ============================================================
# FONCTION PRINCIPALE
# ============================================================
def run_classification_tendance():
    print("=" * 70)
    print("RANDOM FOREST - PREDICTION DES TENDANCES (OPTIMISÉE)")
    print("=" * 70)

    # 1. CHARGEMENT
    print("\n1. Chargement des données...")
    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long", "annee"])
    accidents["annee"] = accidents["annee"].astype(int)
    density["annee"] = pd.to_datetime(density["time"]).dt.year

    # 2. ZONES
    print("\n2. Création des zones...")
    accidents["zone"] = calculer_zone(accidents["lat"], accidents["long"])
    accidents_par_zone = accidents.groupby(["zone", "annee"]).size().reset_index(name="nb_accidents")
    zones = accidents_par_zone["zone"].unique()
    print(f"{len(zones)} zones")

    # 3. DONNÉES COMPLÈTES
    print("\n3. Préparation...")
    annees_historiques = list(range(2011, 2026))
    index_complet = pd.MultiIndex.from_product([zones, annees_historiques], names=["zone", "annee"])
    data = accidents_par_zone.set_index(["zone", "annee"]).reindex(index_complet, fill_value=0).reset_index()
    data["nb_accidents"] = data["nb_accidents"].astype(float)

    # 4. RÉFÉRENCE + CIBLE
    data["reference_5_ans"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
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
    print("\n4. Création des variables...")
    for lag in [1, 2, 3]:
        data[f"nb_accidents_lag_{lag}"] = data.groupby("zone")["nb_accidents"].shift(lag)
    data["moyenne_3"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean()
    )
    data["moyenne_5"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    data["evolution_recente"] = np.where(
        data["moyenne_5"] > 0, ((data["moyenne_3"] - data["moyenne_5"]) / data["moyenne_5"]) * 100, 0
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

    # Flotte (reste global, pas de coordonnées disponibles pour la répartir par zone)
    fleet_total = fleet[fleet["type_navire"] == "Flotte totale"][["annee", "nombre_navires"]].drop_duplicates("annee")
    data = data.merge(fleet_total, on="annee", how="left")
    data["nombre_navires"] = data["nombre_navires"].ffill().bfill()

    # 6. ENTRAÎNEMENT
    print("\n5. Entraînement...")
    FEATURES = ["annee", "densite_moyenne", "densite_max", "nombre_navires",
                "nb_accidents_lag_1", "nb_accidents_lag_2", "nb_accidents_lag_3",
                "moyenne_3", "moyenne_5", "evolution_recente"]

    data_modele = data.dropna(subset=FEATURES + ["categorie"]).copy()
    train = data_modele[data_modele["annee"] <= 2022]
    X_train, y_train = train[FEATURES], train["categorie"]

    print("Répartition des catégories dans l'entraînement :")
    print(y_train.value_counts())

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"Modèle entraîné sur {len(train)} lignes")

    classes_modele = model.classes_  # ordre réel des colonnes de predict_proba

    # 7. PRÉDICTIONS
    print("\n6. Prédictions 2023-2030...")
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

            lag1, lag2, lag3 = accidents_list[-1], accidents_list[-2], accidents_list[-3]
            moyenne_3 = np.mean(accidents_list[-3:])
            moyenne_5 = np.mean(accidents_list[-5:])
            evolution_recente = ((moyenne_3 - moyenne_5) / moyenne_5 * 100) if moyenne_5 > 0 else 0

            X_future = pd.DataFrame(
                [[annee, densite_list[-1], densite_max_list[-1], navires_list[-1],
                  lag1, lag2, lag3, moyenne_3, moyenne_5, evolution_recente]],
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

            # --------------------------------------------------
            # Mise à jour pour l'année suivante
            # --------------------------------------------------
            if annee <= 2025:
                # Année réellement connue -> vraie valeur, pas une estimation
                ligne_reelle = data[(data["zone"] == zone) & (data["annee"] == annee)]
                nb = float(ligne_reelle["nb_accidents"].iloc[0]) if len(ligne_reelle) > 0 else 0.0
            else:
                delta_pct_pondere = sum(
                    proba * DELTA_REPRESENTATIF[classe]
                    for classe, proba in zip(classes_modele, probabilites)
                )
                nb = max(0.0, moyenne_5 * (1 + delta_pct_pondere / 100))

            accidents_list.append(nb)
            densite_list.append(densite_list[-1])
            densite_max_list.append(densite_max_list[-1])
            navires_list.append(navires_list[-1])

    # 8. SAUVEGARDE
    df_futur = pd.DataFrame(resultats)
    output_file = DATA_PATH / "predictions_tendance_2023_2030.csv"
    df_futur.to_csv(output_file, index=False)
    print(f"\n CSV généré: {output_file} ({len(df_futur)} prédictions)")

    return df_futur


run_classification_tendance()


# Si on met toutes les années on a x et si on mets toutes les années à partir de 2014 on a y. 
# Or on remarque que la différence est minime entre les 2 meme pas 1% de différence.