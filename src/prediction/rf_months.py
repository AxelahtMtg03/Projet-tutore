import numpy as np
import pandas as pd
from pathlib import Path

from rf_commun import (
    INSUFFISANT, DISTANCE_MAX_M,
    calculer_zone, categoriser_colonnes, assigner_zones_densite,
    EnsembleSousClasses, afficher_distribution,
    cross_validation_temporelle, afficher_cv, metriques_cv,
    resumer_comparaison, sauvegarder_csv,
    configurer_seuils, calculer_reference_saisonniere, optimiser_poids_classes,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
DATA_PATH.mkdir(parents=True, exist_ok=True)

GRANULARITE = "mensuel"
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
ANNEE_DEBUT = 2014               # premiere annee de l'historique (avant : premier mois des donnees)
DERNIERE_ANNEE_OBSERVEE = 2022   # le modele ne voit RIEN apres decembre de cette annee
ANNEES_TEST = [2023, 2024, 2025]
FIN_PREDICTION = "2030-12-01"

SEUIL_REFERENCE_MIN = 0.15
N_SOUS_CLASSES = "auto"          # ou un entier (ex. 3) pour forcer C1..C3

FICHIER_PREDICTIONS = DATA_PATH / f"predictions_tendance_mensuelle_subdivision{SUFFIXE}.csv"
FICHIER_COMPARAISON = DATA_PATH / f"comparaison_tendance_mensuelle_subdivision{SUFFIXE}.csv"
FICHIER_METRIQUES = DATA_PATH / f"metriques_tendance_mensuelle_subdivision{SUFFIXE}.csv"
FICHIER_CV = DATA_PATH / f"cv_tendance_mensuelle_subdivision{SUFFIXE}.csv"


def run_classification_tendance_mensuelle_subdivision():
    print("=" * 70)
    print("RANDOM FOREST - PREDICTION MENSUELLE AVEC SUBDIVISION")
    print("=" * 70)

    # ============================================================
    # 1. CHARGEMENT
    # ============================================================
    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long"])
    # 1er jour du mois (NaT conserve si la date est invalide)
    accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    accidents = accidents.dropna(subset=["mois"])

    density["mois"] = pd.to_datetime(density["time"]).dt.to_period("M").dt.to_timestamp()

    # ============================================================
    # 2. ZONES
    # ============================================================
    accidents, zones_natives = assigner_zones_densite(accidents, density, DISTANCE_MAX_M)

    # ============================================================
    # 3. AGREGATION
    # ============================================================
    accidents_par_zone_mois = accidents.groupby(["zone", "mois"]).size().reset_index(name="nb_accidents")
    zones = accidents_par_zone_mois["zone"].unique()

    mois_min = max(accidents_par_zone_mois["mois"].min(), pd.Timestamp(f"{ANNEE_DEBUT}-01-01"))
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

    # Reference ALIGNEE SUR LE TEST : mediane des comptes non nuls de la meme (zone, mois de
    # l'annee) sur les annees connues a la date de la cible (jamais apres DERNIERE_ANNEE_OBSERVEE)
    agg_mois = accidents_par_zone_mois.assign(
        annee=accidents_par_zone_mois["mois"].dt.year,
        mois_num=accidents_par_zone_mois["mois"].dt.month)
    refs = calculer_reference_saisonniere(
        agg_mois, ["mois_num"],
        range(ANNEE_DEBUT, int(FIN_PREDICTION[:4]) + 1), DERNIERE_ANNEE_OBSERVEE)
    data = data.merge(refs, on=["zone", "annee", "mois_num"], how="left")
    ref_map = {(z, a, m): (r, n) for z, a, m, r, n in
               zip(refs["zone"], refs["annee"], refs["mois_num"], refs["ref_hist"], refs["nb_annees_hist"])}
    data["accidents_suivant"] = data.groupby("zone")["nb_accidents"].shift(-1)
    data["reference_suivante"] = data.groupby("zone")["ref_hist"].shift(-1)
    data["ref_hist_cible"] = data["reference_suivante"].fillna(0)
    data["nb_annees_hist_cible"] = data.groupby("zone")["nb_annees_hist"].shift(-1).fillna(0)
    data["ecart_absolu_suivant"] = data["accidents_suivant"] - data["reference_suivante"]
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
                "mediane_3", "mediane_6", "mediane_12", "mediane_24", "evolution_recente", "ref_hist_cible", "nb_annees_hist_cible"]

    # ============================================================
    # 6. JEU D'ENTRAINEMENT
    # ============================================================
    # Une ligne "mois m" a pour cible la tendance de m+1. Pour ne rien savoir de
    # 2023+, on garde les lignes dont la cible est <= dec. 2022, donc m < dec. 2022.
    dernier_mois_observe = pd.Timestamp(f"{DERNIERE_ANNEE_OBSERVEE}-12-01")

    data_modele = data.dropna(subset=FEATURES + ["categorie"]).copy()
    data_modele = data_modele[data_modele["categorie"] != INSUFFISANT]
    train = data_modele[data_modele["mois"] < dernier_mois_observe].reset_index(drop=True)

    print(f"\nLignes d'entrainement : {len(train)} "
          f"(mois {train['mois'].min().date()} -> {train['mois'].max().date()})")
    afficher_distribution(train)

    # ============================================================
    # 7. CROSS-VALIDATION TEMPORELLE (1 fold = 1 annee de validation)
    # ============================================================
    params_modele = dict(n_sous_classes=N_SOUS_CLASSES)
    cv_df, oof = cross_validation_temporelle(train, FEATURES, "categorie",
                                        colonne_temps="mois", colonne_groupe="annee",
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
    mois_futurs = pd.date_range(dernier_mois_observe + pd.DateOffset(months=1), FIN_PREDICTION, freq="MS")
    dernier_mois_connu = mois_max

    # Historique limite a dec. 2022 : les mois suivants ne servent qu'a
    # reconstituer le "reel" quand on avance mois par mois.
    historique = data[data["mois"] <= dernier_mois_observe][
        ["zone", "mois", "nb_accidents", "densite_moyenne", "densite_max", "nombre_navires"]
    ]
    reel = {(z, m): n for z, m, n in zip(data["zone"], data["mois"], data["nb_accidents"])}

    accidents_test = accidents[accidents["annee"].between(min(ANNEES_TEST), max(ANNEES_TEST))]
    zones_avec_accidents = set(accidents_test["zone"].unique())

    etat_zones = {}
    for zone, zone_data in historique.groupby("zone"):
        zone_data = zone_data.sort_values("mois")
        etat_zones[zone] = {
            "accidents": zone_data["nb_accidents"].fillna(0).tolist(),
            "densite": zone_data["densite_moyenne"].fillna(0).tolist(),
            "densite_max": zone_data["densite_max"].fillna(0).tolist(),
            "navires": zone_data["nombre_navires"].fillna(0).tolist(),
        }

    zones_valides = [z for z in etat_zones
                     if z in zones_avec_accidents and len(etat_zones[z]["accidents"]) >= 5]
    print(f"{len(zones_valides)}/{len(zones)} zones avec assez d'historique et accidents")

    resultats = []
    for mois in mois_futurs:
        # mois "observe" = le mois juste avant (a l'entrainement, la ligne m porte
        # les features de m et la cible de m+1)
        mois_obs = mois - pd.DateOffset(months=1)
        mois_sin = np.sin(2 * np.pi * mois_obs.month / 12)
        mois_cos = np.cos(2 * np.pi * mois_obs.month / 12)

        lignes_features = []
        for zone in zones_valides:
            s = etat_zones[zone]
            acc = s["accidents"]
            mediane_3 = np.median(acc[-3:])
            mediane_6 = np.median(acc[-6:])
            mediane_12 = np.median(acc[-12:])
            mediane_24 = np.median(acc[-24:])
            evolution_recente = ((mediane_3 - mediane_24) / mediane_24 * 100) if mediane_24 > 0 else 0

            lignes_features.append([
                mois_obs.year, mois_sin, mois_cos, s["densite"][-1], s["densite_max"][-1], s["navires"][-1],
                acc[-1], acc[-2], acc[-3], acc[-6], acc[-12], acc[-24],
                mediane_3, mediane_6, mediane_12, mediane_24, evolution_recente,
                *ref_map.get((zone, mois.year, mois.month), (0.0, 0))
            ])

        probas = model.predict_proba(pd.DataFrame(lignes_features, columns=FEATURES))

        for i, zone in enumerate(zones_valides):
            proba = probas[i]
            idx = int(np.argmax(proba * model.poids_))

            resultats.append({
                "zone": zone,
                "mois": mois.strftime("%Y-%m"),
                "annee": mois.year,
                "mois_num": mois.month,
                "tendance": classes_modele[idx],
                "confiance": round(proba[idx] * 100, 1),
                **{f"proba_{c}": round(p * 100, 1) for c, p in zip(classes_modele, proba)},
            })

            s = etat_zones[zone]
            if mois <= dernier_mois_connu:
                nb = float(reel.get((zone, mois), 0.0))
            else:
                nb = max(0.0, np.median(s["accidents"][-24:]))

            s["accidents"].append(nb)
            s["densite"].append(s["densite"][-1])
            s["densite_max"].append(s["densite_max"][-1])
            s["navires"].append(s["navires"][-1])

    df_futur = pd.DataFrame(resultats).sort_values(["zone", "mois"]).reset_index(drop=True)
    print(f"\nPredictions generees : {len(df_futur)} ({df_futur['zone'].nunique()} zones)")
    sauvegarder_csv(df_futur, FICHIER_PREDICTIONS)

    return df_futur, cv_df


def test_accuracy_subdivision(df_pred=None, cv_df=None):
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
    accidents["mois"] = accidents["mois"].astype(int)

    accidents_par_zone = accidents.groupby(["zone", "annee", "mois"]).size().reset_index(name="nb_accidents")

    ref_zone = (accidents_par_zone[accidents_par_zone["annee"] <= DERNIERE_ANNEE_OBSERVEE]
                .groupby(["zone", "mois"])["nb_accidents"].median().reset_index())
    ref_zone.columns = ["zone", "mois", "reference_historique"]

    toutes_zones = sorted(df_pred["zone"].unique())
    index_complet = pd.MultiIndex.from_product(
        [toutes_zones, ANNEES_TEST, list(range(1, 13))],
        names=["zone", "annee", "mois"]
    )
    accidents_par_zone = (accidents_par_zone
                          .set_index(["zone", "annee", "mois"])
                          .reindex(index_complet, fill_value=0)
                          .reset_index())

    data = accidents_par_zone.merge(ref_zone, on=["zone", "mois"], how="left")
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

    df_pred_eval = df_pred.copy()
    # "mois" est "YYYY-MM" (texte) -> on garde le numero du mois
    if pd.api.types.is_numeric_dtype(df_pred_eval["mois"]):
        df_pred_eval["mois"] = df_pred_eval["mois"].astype(int)
    else:
        df_pred_eval["mois"] = df_pred_eval["mois"].astype(str).str.split("-").str[1].astype(int)

    df_compare = df_pred_eval.merge(data[["zone", "annee", "mois", "tendance_reelle"]],
                                    on=["zone", "annee", "mois"], how="inner")
    df_compare = df_compare[df_compare["tendance_reelle"] != INSUFFISANT]

    if len(df_compare) == 0:
        print("Aucune correspondance")
        return

    df_compare["correct"] = df_compare["tendance"] == df_compare["tendance_reelle"]

    metriques = resumer_comparaison(
        df_compare, GRANULARITE, "ACCURACY - MODELE MENSUEL AVEC SUBDIVISION", "annee"
    )
    if cv_df is not None and not cv_df.empty:
        metriques = pd.concat([metriques, metriques_cv(cv_df, GRANULARITE)], ignore_index=True)

    print()
    sauvegarder_csv(df_compare, FICHIER_COMPARAISON)
    sauvegarder_csv(metriques, FICHIER_METRIQUES)

    return df_compare


def run():
    df_futur, cv_df = run_classification_tendance_mensuelle_subdivision()
    return test_accuracy_subdivision(df_futur, cv_df)


run()