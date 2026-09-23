import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer

from rf_commun import (
    CLASSES, DISTANCE_MAX_M, INSUFFISANT, DELTA_REPRESENTATIF, assigner_zones_densite,
    calculer_zone, categoriser, EnsembleSousClasses,
    cross_validation_temporelle, afficher_cv, metriques_cv,
    resumer_comparaison, sauvegarder_csv,
    configurer_seuils, calculer_reference_saisonniere, optimiser_poids_classes,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
DATA_PATH.mkdir(parents=True, exist_ok=True)

GRANULARITE = "annuel"
# --- Arguments CLI pour le grid search ---
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--seuil-ecart", type=float, default=None)
_parser.add_argument("--seuil-pct", type=float, default=None)
_args = _parser.parse_args()
configurer_seuils(_args.seuil_ecart, _args.seuil_pct)

# Suffixe pour ne pas ecraser les fichiers entre combinaisons
if _args.seuil_ecart is not None and _args.seuil_pct is not None:
    SUFFIXE = f"_e{_args.seuil_ecart}_p{_args.seuil_pct}"
else:
    SUFFIXE = ""
ANNEE_DEBUT = 2014               # premiere annee de l'historique (avant : 2011)
ANNEE_FIN_DONNEES = 2025         # derniere annee de donnees reelles
DERNIERE_ANNEE_OBSERVEE = 2022   # le modele ne voit RIEN apres cette annee
ANNEES_TEST = [2023, 2024, 2025]
ANNEE_FIN_PREDICTION = 2030

N_SOUS_CLASSES = "auto"          # ou un entier (ex. 3) pour forcer C1..C3

FICHIER_PREDICTIONS = DATA_PATH / f"predictions_tendance_annuelle_subdivision{SUFFIXE}.csv"
FICHIER_COMPARAISON = DATA_PATH / f"comparaison_tendance_annuelle_subdivision{SUFFIXE}.csv"
FICHIER_METRIQUES = DATA_PATH / f"metriques_tendance_annuelle_subdivision{SUFFIXE}.csv"
FICHIER_CV = DATA_PATH / f"cv_tendance_annuelle_subdivision{SUFFIXE}.csv"


def run_classification_tendance_annuelle_subdivision():
    print("=" * 70)
    print("RANDOM FOREST - PREDICTION ANNUELLE AVEC SUBDIVISION")
    print("=" * 70)

    # ============================================================
    # 1. CHARGEMENT
    # ============================================================
    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long", "annee"])
    accidents["annee"] = accidents["annee"].astype(int)
    density["annee"] = pd.to_datetime(density["time"]).dt.year

    # ============================================================
    # 2. ZONES
    # ============================================================
    accidents, zones_natives = assigner_zones_densite(accidents, density, DISTANCE_MAX_M)
    accidents_par_zone = accidents.groupby(["zone", "annee"]).size().reset_index(name="nb_accidents")
    zones = accidents_par_zone["zone"].unique()

    annees_historiques = list(range(ANNEE_DEBUT, ANNEE_FIN_DONNEES + 1))
    index_complet = pd.MultiIndex.from_product([zones, annees_historiques], names=["zone", "annee"])
    data = accidents_par_zone.set_index(["zone", "annee"]).reindex(index_complet, fill_value=0).reset_index()
    data["nb_accidents"] = data["nb_accidents"].astype(float)

    # ============================================================
    # 3. REFERENCE ET CIBLE
    # ============================================================
    # Reference ALIGNEE SUR LE TEST : mediane des comptes non nuls de la zone sur les annees
    # connues a la date de la cible (jamais apres DERNIERE_ANNEE_OBSERVEE)
    refs = calculer_reference_saisonniere(
        accidents_par_zone, [], range(ANNEE_DEBUT, ANNEE_FIN_PREDICTION + 1), DERNIERE_ANNEE_OBSERVEE)
    data = data.merge(refs, on=["zone", "annee"], how="left")
    ref_map = {(z, a): (r, n) for z, a, r, n in
               zip(refs["zone"], refs["annee"], refs["ref_hist"], refs["nb_annees_hist"])}
    data["accidents_suivants"] = data.groupby("zone")["nb_accidents"].shift(-1)
    data["reference_suivante"] = data.groupby("zone")["ref_hist"].shift(-1)
    data["ref_hist_cible"] = data["reference_suivante"].fillna(0)
    data["nb_annees_hist_cible"] = data.groupby("zone")["nb_annees_hist"].shift(-1).fillna(0)
    data["ecart_absolu_suivant"] = data["accidents_suivants"] - data["reference_suivante"]
    data["evolution_suivante"] = np.where(
        data["reference_suivante"] > 0.5,
        (data["ecart_absolu_suivant"] / data["reference_suivante"]) * 100,
        np.nan
    )

    # ============================================================
    # 4. CATEGORISATION EN 3 CLASSES
    # ============================================================
    data["categorie"] = data.apply(
        lambda r: categoriser(r["ecart_absolu_suivant"], r["evolution_suivante"], r["reference_suivante"]),
        axis=1
    )

    print("\nDistribution des categories :")
    print(data["categorie"].value_counts(dropna=False))

    # ============================================================
    # 5. FEATURES
    # ============================================================
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
                "mediane_3", "mediane_5", "evolution_recente", "ref_hist_cible", "nb_annees_hist_cible"]

    # ============================================================
    # 6. JEU D'ENTRAINEMENT
    # ============================================================
    # Une ligne "annee = N" a pour cible la tendance de N+1. Pour ne rien
    # savoir de 2023+, on garde les lignes dont la cible est <= 2022,
    # donc annee <= 2021.
    data_modele = data.dropna(subset=FEATURES + ["categorie"]).copy()
    data_modele = data_modele[data_modele["categorie"] != INSUFFISANT]
    train = data_modele[data_modele["annee"] < DERNIERE_ANNEE_OBSERVEE].reset_index(drop=True)

    print(f"\nLignes d'entrainement : {len(train)} "
          f"(annees {train['annee'].min()} -> {train['annee'].max()}, "
          f"cibles {train['annee'].min() + 1} -> {train['annee'].max() + 1})")
    print("\nDistribution de la cible (3 classes) :")
    comptes = train["categorie"].value_counts()
    for classe in CLASSES:
        n = int(comptes.get(classe, 0))
        print(f"  {classe}: {n} ({n / len(train) * 100:.1f}%)")

    # ============================================================
    # 7. CROSS-VALIDATION TEMPORELLE
    # ============================================================
    params_modele = dict(n_sous_classes=N_SOUS_CLASSES)
    cv_df, oof = cross_validation_temporelle(train, FEATURES, "categorie", "annee", avec_oof=True, **params_modele)
    afficher_cv(cv_df)
    if not cv_df.empty:
        sauvegarder_csv(cv_df, FICHIER_CV)

    # ============================================================
    # 8. ENTRAINEMENT FINAL (ENSEMBLE PAR SOUS-CLASSES DE STABLE)
    # ============================================================
    print("\nEntrainement du modele final...")
    model = EnsembleSousClasses(verbose=True, **params_modele)
    model.fit(train[FEATURES], train["categorie"])
    classes_modele = list(model.classes_)
    model.poids_ = optimiser_poids_classes(oof)
    print(f"{len(model.modeles_)} modeles entraines (moyenne de leurs probabilites)")

    # ============================================================
    # 9. PREDICTIONS
    # ============================================================
    # Historique limite a 2022 : les annees 2023+ ne servent qu'a reconstituer
    # le "reel" quand on avance annee par annee.
    historique = data[data["annee"] <= DERNIERE_ANNEE_OBSERVEE][
        ["zone", "annee", "nb_accidents", "densite_moyenne", "densite_max", "nombre_navires"]
    ]
    reel = data.set_index(["zone", "annee"])["nb_accidents"].to_dict()

    # Zones valides = >= 1 accident entre 2023 et 2025 (niveau annuel)
        # Zones valides = >= 1 accident entre 2023 et 2025 (meme filtre que les autres granularites)
    accidents_test = accidents[accidents["annee"].between(min(ANNEES_TEST), max(ANNEES_TEST))]
    zones_avec_accidents = set(accidents_test["zone"].unique())

    etat = {}
    for zone, zone_data in historique.groupby("zone"):
        if zone not in zones_avec_accidents or len(zone_data) < 5:
            continue
        zone_data = zone_data.sort_values("annee")
        etat[zone] = {
            "acc": zone_data["nb_accidents"].fillna(0).tolist(),
            "dens": zone_data["densite_moyenne"].fillna(0).tolist(),
            "dens_max": zone_data["densite_max"].fillna(0).tolist(),
            "nav": zone_data["nombre_navires"].fillna(0).tolist(),
        }

    resultats = []
    for annee in range(DERNIERE_ANNEE_OBSERVEE + 1, ANNEE_FIN_PREDICTION + 1):
        # Features de toutes les zones pour cette annee (prediction en lot)
        zones_annee, lignes, medianes_5 = [], [], []
        for zone, e in etat.items():
            acc = e["acc"]
            mediane_3 = np.median(acc[-3:])
            mediane_5 = np.median(acc[-5:])
            evolution_recente = ((mediane_3 - mediane_5) / mediane_5 * 100) if mediane_5 > 0 else 0
            # "annee" = derniere annee observee (comme a l'entrainement, ou la cible est N+1)
            lignes.append([annee - 1, e["dens"][-1], e["dens_max"][-1], e["nav"][-1],
                           acc[-1], acc[-2], acc[-3], acc[-4], acc[-5],
                           mediane_3, mediane_5, evolution_recente,
                           *ref_map.get((zone, annee), (0.0, 0))])
            zones_annee.append(zone)
            medianes_5.append(mediane_5)

        probas = model.predict_proba(pd.DataFrame(lignes, columns=FEATURES))

        for k, zone in enumerate(zones_annee):
            proba = probas[k]
            idx = int(np.argmax(proba * model.poids_))
            e = etat[zone]

            resultats.append({
                "zone": zone,
                "annee": annee,
                "tendance": classes_modele[idx],
                "confiance": round(proba[idx] * 100, 1),
                **{f"proba_{c}": round(p * 100, 1) for c, p in zip(classes_modele, proba)},
            })

            if annee <= ANNEE_FIN_DONNEES:
                nb = float(reel.get((zone, annee), 0.0))
            else:
                delta_pct_pondere = sum(
                    p * DELTA_REPRESENTATIF.get(c, 0) for c, p in zip(classes_modele, proba)
                )
                nb = max(0.0, medianes_5[k] * (1 + delta_pct_pondere / 100))

            e["acc"].append(nb)
            e["dens"].append(e["dens"][-1])
            e["dens_max"].append(e["dens_max"][-1])
            e["nav"].append(e["nav"][-1])

    df_futur = pd.DataFrame(resultats).sort_values(["zone", "annee"]).reset_index(drop=True)
    print(f"\nPredictions generees : {len(df_futur)} ({df_futur['zone'].nunique()} zones)")
    sauvegarder_csv(df_futur, FICHIER_PREDICTIONS)

    return df_futur, cv_df


def test_accuracy_annuel_subdivision(df_pred=None, cv_df=None):
    if df_pred is None:
        try:
            df_pred = pd.read_csv(FICHIER_PREDICTIONS, encoding="utf-8-sig")
        except FileNotFoundError:
            print("Fichier de predictions non trouve")
            return

    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    accidents = accidents.dropna(subset=["lat", "long", "annee"])
    accidents["annee"] = accidents["annee"].astype(int)

    accidents["zone"] = calculer_zone(accidents["lat"], accidents["long"])
    accidents_par_zone = accidents.groupby(["zone", "annee"]).size().reset_index(name="nb_accidents")

    # Reference : mediane historique de la zone jusqu'a 2022 (calculée AVANT reindex)
    ref_zone = (accidents_par_zone[accidents_par_zone["annee"] <= DERNIERE_ANNEE_OBSERVEE]
                .groupby("zone")["nb_accidents"].median().reset_index())
    ref_zone.columns = ["zone", "reference_historique"]

    # Maintenant le reindex sur les zones du modele x annees de test
    toutes_zones = sorted(df_pred["zone"].unique())
    index_complet = pd.MultiIndex.from_product(
        [toutes_zones, ANNEES_TEST],
        names=["zone", "annee"]
    )
    accidents_par_zone = (accidents_par_zone
                          .set_index(["zone", "annee"])
                          .reindex(index_complet, fill_value=0)
                          .reset_index())

    data = accidents_par_zone.merge(ref_zone, on="zone", how="left")
    data["reference_historique"] = data["reference_historique"].fillna(0)

    data["ecart_absolu"] = data["nb_accidents"] - data["reference_historique"]
    with np.errstate(divide="ignore", invalid="ignore"):
        data["evolution"] = np.where(
            data["reference_historique"] > 0,
            data["ecart_absolu"] / data["reference_historique"] * 100,
            0.0
        )
    data["tendance_reelle"] = [
        categoriser(e, p, r)
        for e, p, r in zip(data["ecart_absolu"], data["evolution"], data["reference_historique"])
    ]
    data = data[data["annee"].isin(ANNEES_TEST)]

    # === DEBUG : afficher les types et exemples avant le merge ===
    # print("df_pred dtypes :", df_pred[["zone", "annee"]].dtypes.to_dict())
    # print("data dtypes    :", data[["zone", "annee"]].dtypes.to_dict())
    # print("df_pred ex     :", df_pred[["zone", "annee"]].head(3).to_dict("records"))
    # print("data ex        :", data[["zone", "annee"]].head(3).to_dict("records"))
    # print("Zones df_pred  :", sorted(df_pred["zone"].unique())[:3])
    # print("Zones data     :", sorted(data["zone"].unique())[:3])
    # print("Intersection   :", len(set(df_pred["zone"]) & set(data["zone"])))

    df_compare = df_pred.merge(data[["zone", "annee", "tendance_reelle"]], on=["zone", "annee"], how="inner")
    df_compare = df_compare[df_compare["tendance_reelle"] != INSUFFISANT]

    if len(df_compare) == 0:
        print("Aucune correspondance")
        return

    df_compare["correct"] = df_compare["tendance"] == df_compare["tendance_reelle"]

    metriques = resumer_comparaison(
        df_compare, GRANULARITE, "ACCURACY - MODELE ANNUEL AVEC SUBDIVISION", "annee"
    )
    if cv_df is not None and not cv_df.empty:
        metriques = pd.concat([metriques, metriques_cv(cv_df, GRANULARITE)], ignore_index=True)

    print()
    sauvegarder_csv(df_compare, FICHIER_COMPARAISON)
    sauvegarder_csv(metriques, FICHIER_METRIQUES)

    return df_compare


def run():
    df_futur, cv_df = run_classification_tendance_annuelle_subdivision()
    return test_accuracy_annuel_subdivision(df_futur, cv_df)


run()