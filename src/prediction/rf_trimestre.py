import numpy as np
import pandas as pd
from pathlib import Path

from rf_commun import (
    INSUFFISANT, STABLE, DIMINUTION, AUGMENTATION, DISTANCE_MAX_M,
    calculer_zone, categoriser_colonnes, assigner_zones_densite,
    EnsembleSousClasses, afficher_distribution,
    cross_validation_temporelle, afficher_cv, metriques_cv,
    resumer_comparaison, sauvegarder_csv,
    configurer_seuils, calculer_reference_saisonniere, optimiser_poids_classes,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
DATA_PATH.mkdir(parents=True, exist_ok=True)

GRANULARITE = "trimestriel"
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
DERNIERE_ANNEE_OBSERVEE = 2022   # le modele ne voit RIEN apres le T4 de cette annee
ANNEES_TEST = [2023, 2024, 2025]
ANNEE_FIN_PREDICTION = 2030

SEUIL_REFERENCE_MIN = 0.15
N_SOUS_CLASSES = "auto"          # ou un entier (ex. 3) pour forcer C1..C3

# Variation representative (avant : faible = +/-10, forte = +/-30 -> milieu = 20)
DELTA_REPRESENTATIF = {DIMINUTION: -20, STABLE: 0, AUGMENTATION: 20, INSUFFISANT: 0}

FICHIER_PREDICTIONS = DATA_PATH / f"predictions_tendance_trimestrielle_subdivision{SUFFIXE}.csv"
FICHIER_COMPARAISON = DATA_PATH / f"comparaison_tendance_trimestrielle_subdivision{SUFFIXE}.csv"
FICHIER_METRIQUES = DATA_PATH / f"metriques_tendance_trimestrielle_subdivision{SUFFIXE}.csv"
FICHIER_CV = DATA_PATH / f"cv_tendance_trimestrielle_subdivision{SUFFIXE}.csv"


def run_classification_tendance_trimestrielle_subdivision():
    print("=" * 70)
    print("RANDOM FOREST - PREDICTION TRIMESTRIELLE AVEC SUBDIVISION")
    print("=" * 70)

    # ============================================================
    # 1. CHARGEMENT
    # ============================================================
    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long", "annee"])
    accidents["annee"] = accidents["annee"].astype(int)

    accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"], errors="coerce").dt.month
    accidents = accidents.dropna(subset=["mois"])
    accidents["trimestre"] = np.ceil(accidents["mois"] / 3).astype(int)

    density["mois"] = pd.to_datetime(density["time"]).dt.month
    density["trimestre"] = np.ceil(density["mois"] / 3).astype(int)
    density["annee"] = pd.to_datetime(density["time"]).dt.year

    # ============================================================
    # 2. ZONES
    # ============================================================
    accidents, zones_natives = assigner_zones_densite(accidents, density, DISTANCE_MAX_M)

    # ============================================================
    # 3. AGREGATION
    # ============================================================
    accidents_par_zone_trimestre = accidents.groupby(["zone", "annee", "trimestre"]).size().reset_index(name="nb_accidents")
    zones = accidents_par_zone_trimestre["zone"].unique()

    index_complet = pd.MultiIndex.from_product(
        [zones, range(ANNEE_DEBUT, ANNEE_FIN_DONNEES + 1), [1, 2, 3, 4]],
        names=["zone", "annee", "trimestre"]
    )

    data = accidents_par_zone_trimestre.set_index(["zone", "annee", "trimestre"]).reindex(index_complet, fill_value=0).reset_index()
    data["nb_accidents"] = data["nb_accidents"].astype(float)
    # index chronologique du trimestre (0 = T1 de ANNEE_DEBUT)
    data["periode_idx"] = (data["annee"] - ANNEE_DEBUT) * 4 + (data["trimestre"] - 1)

    # Reference ALIGNEE SUR LE TEST : mediane des comptes non nuls de la meme (zone, trimestre)
    # sur les annees connues a la date de la cible (jamais apres DERNIERE_ANNEE_OBSERVEE)
    refs = calculer_reference_saisonniere(
        accidents_par_zone_trimestre, ["trimestre"],
        range(ANNEE_DEBUT, ANNEE_FIN_PREDICTION + 1), DERNIERE_ANNEE_OBSERVEE)
    data = data.merge(refs, on=["zone", "annee", "trimestre"], how="left")
    ref_map = {(z, a, t): (r, n) for z, a, t, r, n in
               zip(refs["zone"], refs["annee"], refs["trimestre"], refs["ref_hist"], refs["nb_annees_hist"])}
    data["accidents_suivants"] = data.groupby("zone")["nb_accidents"].shift(-1)
    data["reference_suivante"] = data.groupby("zone")["ref_hist"].shift(-1)
    data["ref_hist_cible"] = data["reference_suivante"].fillna(0)
    data["nb_annees_hist_cible"] = data.groupby("zone")["nb_annees_hist"].shift(-1).fillna(0)
    data["ecart_absolu_suivant"] = data["accidents_suivants"] - data["reference_suivante"]
    data["evolution_suivante"] = np.where(
        data["reference_suivante"] > SEUIL_REFERENCE_MIN,
        (data["ecart_absolu_suivant"] / data["reference_suivante"]) * 100,
        np.nan
    )

    # ============================================================
    # 4. CATEGORISATION EN 3 CLASSES
    # ============================================================
    data["categorie"] = categoriser_colonnes(
        data["ecart_absolu_suivant"], data["evolution_suivante"], data["reference_suivante"]
    )

    print("\nDistribution des categories :")
    print(data["categorie"].value_counts(dropna=False))

    # ============================================================
    # 5. FEATURES
    # ============================================================
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
                "mediane_4", "mediane_8", "mediane_12", "mediane_20", "evolution_recente", "ref_hist_cible", "nb_annees_hist_cible"]

    # ============================================================
    # 6. JEU D'ENTRAINEMENT
    # ============================================================
    # Une ligne "periode t" a pour cible la tendance de t+1. Pour ne rien savoir
    # de 2023+, on garde les lignes dont la cible est <= T4 2022, donc t < T4 2022.
    idx_derniere_observee = (DERNIERE_ANNEE_OBSERVEE - ANNEE_DEBUT) * 4 + 3  # T4 de 2022

    data_modele = data.dropna(subset=FEATURES + ["categorie"]).copy()
    data_modele = data_modele[data_modele["categorie"] != INSUFFISANT]
    train = data_modele[data_modele["periode_idx"] < idx_derniere_observee].reset_index(drop=True)

    print(f"\nLignes d'entrainement : {len(train)} "
          f"(annees {train['annee'].min()} -> {train['annee'].max()})")
    afficher_distribution(train)

    # ============================================================
    # 7. CROSS-VALIDATION TEMPORELLE (1 fold = 1 annee de validation)
    # ============================================================
    params_modele = dict(n_sous_classes=N_SOUS_CLASSES)
    cv_df, oof = cross_validation_temporelle(train, FEATURES, "categorie",
                                        colonne_temps="periode_idx", colonne_groupe="annee",
                                        avec_oof=True, **params_modele)
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
    # Historique limite a T4 2022 : les trimestres suivants ne servent qu'a
    # reconstituer le "reel" quand on avance trimestre par trimestre.
    historique = data[data["periode_idx"] <= idx_derniere_observee][
        ["zone", "annee", "trimestre", "nb_accidents", "densite_moyenne", "densite_max", "nombre_navires"]
    ]
    reel = {(z, a, t): n for z, a, t, n in
            zip(data["zone"], data["annee"], data["trimestre"], data["nb_accidents"])}

    accidents_test = accidents[accidents["annee"].between(min(ANNEES_TEST), max(ANNEES_TEST))]
    zones_avec_accidents = set(accidents_test["zone"].unique())

    etat_zones = {}
    for zone, zone_data in historique.groupby("zone"):
        zone_data = zone_data.sort_values(["annee", "trimestre"])
        etat_zones[zone] = {
            "accidents": zone_data["nb_accidents"].fillna(0).tolist(),
            "densite": zone_data["densite_moyenne"].fillna(0).tolist(),
            "densite_max": zone_data["densite_max"].fillna(0).tolist(),
            "navires": zone_data["nombre_navires"].fillna(0).tolist(),
        }

    zones_valides = [z for z in etat_zones
                     if z in zones_avec_accidents and len(etat_zones[z]["accidents"]) >= 5]
    print(f"{len(zones_valides)}/{len(zones)} zones avec assez d'historique et accidents")

    future_trimestres = [(a, t) for a in range(DERNIERE_ANNEE_OBSERVEE + 1, ANNEE_FIN_PREDICTION + 1)
                         for t in [1, 2, 3, 4]]
    # periode "observee" associee a chaque periode predite = la periode juste avant
    # (a l'entrainement, la ligne t porte les features de t et la cible de t+1)
    precedentes = [(DERNIERE_ANNEE_OBSERVEE, 4)] + future_trimestres[:-1]

    resultats = []
    for (annee, trimestre), (annee_obs, trimestre_obs) in zip(future_trimestres, precedentes):
        lignes_features = []
        medianes_20 = []
        for zone in zones_valides:
            s = etat_zones[zone]
            acc = s["accidents"]
            mediane_4 = np.median(acc[-4:])
            mediane_8 = np.median(acc[-8:])
            mediane_12 = np.median(acc[-12:])
            mediane_20 = np.median(acc[-20:])
            evolution_recente = ((mediane_4 - mediane_20) / mediane_20 * 100) if mediane_20 > 0 else 0

            lignes_features.append([
                annee_obs, trimestre_obs, s["densite"][-1], s["densite_max"][-1], s["navires"][-1],
                acc[-1], acc[-2], acc[-3], acc[-4], acc[-8], acc[-12], acc[-16], acc[-20],
                mediane_4, mediane_8, mediane_12, mediane_20, evolution_recente,
                *ref_map.get((zone, annee, trimestre), (0.0, 0))
            ])
            medianes_20.append(mediane_20)

        probas = model.predict_proba(pd.DataFrame(lignes_features, columns=FEATURES))

        for i, zone in enumerate(zones_valides):
            proba = probas[i]
            idx = int(np.argmax(proba * model.poids_))

            resultats.append({
                "zone": zone,
                "annee": annee,
                "trimestre": trimestre,
                "trimestre_annee": f"{annee}-T{trimestre}",
                "tendance": classes_modele[idx],
                "confiance": round(proba[idx] * 100, 1),
                **{f"proba_{c}": round(p * 100, 1) for c, p in zip(classes_modele, proba)},
            })

            s = etat_zones[zone]
            if (zone, annee, trimestre) in reel:
                nb = float(reel[(zone, annee, trimestre)])
            else:
                delta_pct_pondere = sum(
                    p * DELTA_REPRESENTATIF.get(c, 0) for c, p in zip(classes_modele, proba)
                )
                nb = max(0.0, medianes_20[i] * (1 + delta_pct_pondere / 100))

            s["accidents"].append(nb)
            s["densite"].append(s["densite"][-1])
            s["densite_max"].append(s["densite_max"][-1])
            s["navires"].append(s["navires"][-1])

    df_futur = pd.DataFrame(resultats).sort_values(["zone", "annee", "trimestre"]).reset_index(drop=True)
    print(f"\nPredictions generees : {len(df_futur)} ({df_futur['zone'].nunique()} zones)")
    sauvegarder_csv(df_futur, FICHIER_PREDICTIONS)

    return df_futur, cv_df


def test_accuracy_trimestriel_subdivision(df_pred=None, cv_df=None):
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
    accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"], errors="coerce").dt.month
    accidents = accidents.dropna(subset=["mois"])
    accidents["trimestre"] = np.ceil(accidents["mois"] / 3).astype(int)

    accidents_par_zone = accidents.groupby(["zone", "annee", "trimestre"]).size().reset_index(name="nb_accidents")

    ref_zone = (accidents_par_zone[accidents_par_zone["annee"] <= DERNIERE_ANNEE_OBSERVEE]
                .groupby(["zone", "trimestre"])["nb_accidents"].median().reset_index())
    ref_zone.columns = ["zone", "trimestre", "reference_historique"]

    toutes_zones = sorted(df_pred["zone"].unique())
    index_complet = pd.MultiIndex.from_product(
        [toutes_zones, ANNEES_TEST, [1, 2, 3, 4]],
        names=["zone", "annee", "trimestre"]
    )
    accidents_par_zone = (accidents_par_zone
                          .set_index(["zone", "annee", "trimestre"])
                          .reindex(index_complet, fill_value=0)
                          .reset_index())

    data = accidents_par_zone.merge(ref_zone, on=["zone", "trimestre"], how="left")
    data["reference_historique"] = data["reference_historique"].fillna(0)
    
    data["ecart_absolu"] = data["nb_accidents"] - data["reference_historique"]
    with np.errstate(divide="ignore", invalid="ignore"):
        data["evolution"] = np.where(
            data["reference_historique"] > 0,
            data["ecart_absolu"] / data["reference_historique"] * 100,
            0.0
        )
    data["tendance_reelle"] = categoriser_colonnes(
        data["ecart_absolu"], data["evolution"], data["reference_historique"]
    )
    data = data[data["annee"].isin(ANNEES_TEST)]

    df_pred = df_pred.copy()
    df_pred["trimestre"] = df_pred["trimestre"].astype(int)
    data = data.copy()
    data["trimestre"] = data["trimestre"].astype(int)

    df_compare = df_pred.merge(data[["zone", "annee", "trimestre", "tendance_reelle"]],
                               on=["zone", "annee", "trimestre"], how="inner")
    df_compare = df_compare[df_compare["tendance_reelle"] != INSUFFISANT]

    if len(df_compare) == 0:
        print("Aucune correspondance")
        return

    df_compare["correct"] = df_compare["tendance"] == df_compare["tendance_reelle"]

    metriques = resumer_comparaison(
        df_compare, GRANULARITE, "ACCURACY - MODELE TRIMESTRIEL AVEC SUBDIVISION", "annee"
    )
    if cv_df is not None and not cv_df.empty:
        metriques = pd.concat([metriques, metriques_cv(cv_df, GRANULARITE)], ignore_index=True)

    print()
    sauvegarder_csv(df_compare, FICHIER_COMPARAISON)
    sauvegarder_csv(metriques, FICHIER_METRIQUES)

    return df_compare


def run():
    df_futur, cv_df = run_classification_tendance_trimestrielle_subdivision()
    return test_accuracy_trimestriel_subdivision(df_futur, cv_df)


run()