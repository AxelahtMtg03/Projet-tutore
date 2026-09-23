"""
Calcule l'importance des variables du modele mensuel (Random Forest par sous-classes).

Genere :
  - feature_importance.png : graphique horizontal des importances
  - feature_importance.csv : tableau variable / importance

A executer depuis le dossier src/prediction/ :
    python importance_variables.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rf_commun import (
    INSUFFISANT, DISTANCE_MAX_M,
    assigner_zones_densite, categoriser_colonnes,
    EnsembleSousClasses, calculer_reference_saisonniere,
)
import rf_months as rm  # on reutilise les constantes de rf_months

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"


def charger_donnees():
    """Reproduit le pipeline de rf_months.py jusqu'au jeu d'entrainement."""
    accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
    fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
    density = pd.read_csv(DATA_PATH / "vessel_density.csv")

    accidents = accidents.dropna(subset=["lat", "long"])
    accidents["mois"] = pd.to_datetime(
        accidents["Date of occurrence"], errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()
    accidents = accidents.dropna(subset=["mois"])
    density["mois"] = pd.to_datetime(density["time"]).dt.to_period("M").dt.to_timestamp()

    accidents, zones_natives = assigner_zones_densite(accidents, density, DISTANCE_MAX_M)

    accidents_par_zone_mois = accidents.groupby(["zone", "mois"]).size().reset_index(name="nb_accidents")
    zones = accidents_par_zone_mois["zone"].unique()

    mois_min = max(accidents_par_zone_mois["mois"].min(), pd.Timestamp(f"{rm.ANNEE_DEBUT}-01-01"))
    mois_max = accidents_par_zone_mois["mois"].max()
    mois_historiques = pd.date_range(mois_min, mois_max, freq="MS")

    index_complet = pd.MultiIndex.from_product([zones, mois_historiques], names=["zone", "mois"])
    data = accidents_par_zone_mois.set_index(["zone", "mois"]).reindex(index_complet, fill_value=0).reset_index()
    data["nb_accidents"] = data["nb_accidents"].astype(float)
    data["annee"] = data["mois"].dt.year
    data["mois_num"] = data["mois"].dt.month
    data["mois_sin"] = np.sin(2 * np.pi * data["mois_num"] / 12)
    data["mois_cos"] = np.cos(2 * np.pi * data["mois_num"] / 12)

    agg_mois = accidents_par_zone_mois.assign(
        annee=accidents_par_zone_mois["mois"].dt.year,
        mois_num=accidents_par_zone_mois["mois"].dt.month)
    refs = calculer_reference_saisonniere(
        agg_mois, ["mois_num"],
        range(rm.ANNEE_DEBUT, int(rm.FIN_PREDICTION[:4]) + 1),
        rm.DERNIERE_ANNEE_OBSERVEE)
    data = data.merge(refs, on=["zone", "annee", "mois_num"], how="left")

    data["accidents_suivant"] = data.groupby("zone")["nb_accidents"].shift(-1)
    data["reference_suivante"] = data.groupby("zone")["ref_hist"].shift(-1)
    data["ref_hist_cible"] = data["reference_suivante"].fillna(0)
    data["nb_annees_hist_cible"] = data.groupby("zone")["nb_annees_hist"].shift(-1).fillna(0)
    data["ecart_absolu_suivant"] = data["accidents_suivant"] - data["reference_suivante"]
    data["evolution_suivante"] = np.where(
        data["reference_suivante"] > rm.SEUIL_REFERENCE_MIN,
        (data["ecart_absolu_suivant"] / data["reference_suivante"]) * 100,
        np.nan
    )

    data["categorie"] = categoriser_colonnes(
        data["ecart_absolu_suivant"], data["evolution_suivante"], data["reference_suivante"]
    )

    # Lags + medianes
    for lag in [1, 2, 3, 6, 12, 24]:
        data[f"nb_accidents_lag_{lag}"] = data.groupby("zone")["nb_accidents"].shift(lag)
    for w in [3, 6, 12, 24]:
        data[f"mediane_{w}"] = data.groupby("zone")["nb_accidents"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(2, w // 2)).median()
        )
    data["evolution_recente"] = np.where(
        data["mediane_24"] > 0,
        ((data["mediane_3"] - data["mediane_24"]) / data["mediane_24"]) * 100,
        0
    )

    # Densite
    density = density.merge(zones_natives[["latitude", "longitude", "zone"]],
                            on=["latitude", "longitude"], how="left")
    dens = density.groupby(["zone", "mois"]).agg(
        densite_moyenne=("vd", "mean"), densite_max=("vd", "max")).reset_index()
    data = data.merge(dens, on=["zone", "mois"], how="left")
    data[["densite_moyenne", "densite_max"]] = (
        data.groupby("zone")[["densite_moyenne", "densite_max"]].transform(lambda s: s.ffill().bfill())
    )

    # Flotte
    fleet_total = fleet[fleet["type_navire"] == "Flotte totale"][["annee", "nombre_navires"]].drop_duplicates("annee")
    data = data.merge(fleet_total, on="annee", how="left")
    data["nombre_navires"] = data["nombre_navires"].ffill().bfill()

    features = ["annee", "mois_sin", "mois_cos", "densite_moyenne", "densite_max", "nombre_navires",
                "nb_accidents_lag_1", "nb_accidents_lag_2", "nb_accidents_lag_3",
                "nb_accidents_lag_6", "nb_accidents_lag_12", "nb_accidents_lag_24",
                "mediane_3", "mediane_6", "mediane_12", "mediane_24", "evolution_recente",
                "ref_hist_cible", "nb_annees_hist_cible"]

    dernier_mois = pd.Timestamp(f"{rm.DERNIERE_ANNEE_OBSERVEE}-12-01")
    data_modele = data.dropna(subset=features + ["categorie"]).copy()
    data_modele = data_modele[data_modele["categorie"] != INSUFFISANT]
    train = data_modele[data_modele["mois"] < dernier_mois].reset_index(drop=True)
    return train, features


def main():
    print("=" * 70)
    print("IMPORTANCE DES VARIABLES - MODELE MENSUEL")
    print("=" * 70)

    train, features = charger_donnees()
    print(f"Lignes d'entrainement : {len(train)}")

    model = EnsembleSousClasses(verbose=True)
    model.fit(train[features], train["categorie"])
    print(f"{len(model.modeles_)} modeles entraines")

    # Moyenne des importances sur tous les sous-modeles
    importances = np.mean([m.feature_importances_ for m in model.modeles_], axis=0)
    df = pd.DataFrame({"variable": features, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(1)

    print("\nTop 10 variables les plus importantes :")
    print(df.head(10).to_string(index=False))

    # Sauvegarde CSV
    csv_path = DATA_PATH / "feature_importance.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV sauvegarde: {csv_path}")

    # Graphique
    df_plot = df.sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(df_plot["variable"], df_plot["importance"], color="steelblue")
    plt.xlabel("Importance moyenne (Gini)")
    plt.title("Importance des variables - modele mensuel")
    plt.tight_layout()
    png_path = DATA_PATH / "feature_importance.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graphique sauvegarde: {png_path}")


if __name__ == "__main__":
    main()