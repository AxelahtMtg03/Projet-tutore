import os
import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    accuracy_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# CHEMINS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "visualization" / "predictions"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# COULEURS
# ============================================================
COULEURS = {
    "Forte diminution": "#08306b",
    "Faible diminution": "#27ae60",
    "Stable": "#8e44ad",
    "Faible augmentation": "#f4d03f",
    "Forte augmentation": "#e74c3c",
    "Donnees insuffisantes": "#cccccc",
}

# ============================================================
# FONCTION UTILITAIRE : Génération de données MOCK pour test
# ============================================================
def generer_donnees_mock():
    """Génère des données mock si les fichiers réels manquent."""
    print("\n⚠️  Fichiers de données introuvables. Génération de données MOCK pour test...")

    np.random.seed(42)
    n_zones = 100
    annees = list(range(2011, 2026))

    # Accidents mock
    accidents = []
    for _ in range(5000):
        lat = np.random.uniform(30, 70)
        lon = np.random.uniform(-20, 40)
        annee = np.random.choice(annees)
        accidents.append({"lat": lat, "long": lon, "annee": annee})
    accidents = pd.DataFrame(accidents)

    # Densité mock
    density = []
    for _ in range(10000):
        lat = np.random.uniform(30, 70)
        lon = np.random.uniform(-20, 40)
        time = f"{np.random.randint(2011, 2026)}-01-01"
        vd = np.random.uniform(0, 100)
        density.append({"latitude": lat, "longitude": lon, "time": time, "vd": vd})
    density = pd.DataFrame(density)

    # Flotte mock
    fleet = []
    for annee in annees:
        for type_navire in ["Flotte totale", "Cargo", "Pétrolier", "Passager"]:
            fleet.append({
                "annee": annee,
                "type_navire": type_navire,
                "nombre_navires": np.random.randint(1000, 10000)
            })
    fleet = pd.DataFrame(fleet)

    return accidents, density, fleet

# ============================================================
# VÉRIFICATION DES DONNÉES
# ============================================================
def charger_ou_creer_donnees():
    """Charge les données réelles ou génère des mocks."""
    fichiers_manquants = []
    for fichier in ["maritime_accidents.csv", "vessel_density.csv", "global_fleet.csv"]:
        if not (DATA_PATH / fichier).exists():
            fichiers_manquants.append(fichier)

    if fichiers_manquants:
        print(f"\n❌ Fichiers manquants dans {DATA_PATH}: {fichiers_manquants}")
        return generer_donnees_mock()
    else:
        print(f"\n✅ Tous les fichiers trouvés dans {DATA_PATH}")
        accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
        density = pd.read_csv(DATA_PATH / "vessel_density.csv")
        fleet = pd.read_csv(DATA_PATH / "global_fleet.csv")
        return accidents, density, fleet

# ============================================================
# FONCTION 2 : MODÈLE CLASSIFICATION DES TENDANCES (AMÉLIORÉE)
# ============================================================
def run_classification_tendance():
    """
    Random Forest - prédiction de la tendance des accidents.
    Version améliorée avec gestion d'erreurs et logs détaillés.
    """
    print("=" * 70)
    print("RANDOM FOREST - PRÉDICTION DES TENDANCES (AMÉLIORÉ)")
    print("=" * 70)

    # 1. Chargement des données (réelles ou mock)
    try:
        accidents, fleet, density = charger_ou_creer_donnees()
        print(f"\n✅ Données chargées : {len(accidents)} accidents, {len(density)} densités, {len(fleet)} flotte")
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement : {e}")
        return None

    # 2. Nettoyage
    accidents = accidents.dropna(subset=["lat", "long", "annee"])
    accidents["annee"] = accidents["annee"].astype(int)
    density["annee"] = pd.to_datetime(density["time"]).dt.year

    # 3. Création des zones (grille 1.5° x 1.5°)
    def assigner_zone(row):
        lat_zone = np.floor(row["lat"] / 1.5) * 1.5
        lon_zone = np.floor(row["long"] / 1.5) * 1.5
        return f"z_{lat_zone:.1f}_{lon_zone:.1f}"

    accidents["zone"] = accidents.apply(assigner_zone, axis=1)
    accidents_par_zone = (
        accidents.groupby(["zone", "annee"])
        .size()
        .reset_index(name="nb_accidents")
    )
    print(f"\n📍 {accidents_par_zone['zone'].nunique()} zones identifiées")

    # 4. Compléter les années manquantes
    annees_historiques = list(range(2011, 2026))
    zones = accidents_par_zone["zone"].unique()
    index_complet = pd.MultiIndex.from_product([zones, annees_historiques], names=["zone", "annee"])
    data = (
        accidents_par_zone
        .set_index(["zone", "annee"])
        .reindex(index_complet, fill_value=0)
        .reset_index()
    )
    data["nb_accidents"] = data["nb_accidents"].astype(float)

    # 5. Référence historique (moyenne mobile 5 ans)
    data["reference_5_ans"] = (
        data.groupby("zone")["nb_accidents"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    )

    # 6. Création de la cible : tendance de l'année suivante
    data["accidents_suivants"] = data.groupby("zone")["nb_accidents"].shift(-1)
    data["reference_suivante"] = data.groupby("zone")["reference_5_ans"].shift(-1)
    data["evolution_suivante"] = np.where(
        data["reference_suivante"] > 0.5,
        ((data["accidents_suivants"] - data["reference_suivante"]) / data["reference_suivante"]) * 100,
        np.nan
    )

    def categoriser(evolution):
        if pd.isna(evolution):
            return np.nan
        if evolution < -30: return "Forte diminution"
        elif evolution < -10: return "Faible diminution"
        elif evolution < 10: return "Stable"
        elif evolution < 30: return "Faible augmentation"
        else: return "Forte augmentation"

    data["categorie"] = data["evolution_suivante"].apply(categoriser)

    # 7. Features temporelles
    print("\n🔧 Création des features...")
    for lag in [1, 2, 3]:
        data[f"nb_accidents_lag_{lag}"] = data.groupby("zone")["nb_accidents"].shift(lag)

    data["moyenne_3"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean()
    )
    data["moyenne_5"] = data.groupby("zone")["nb_accidents"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    data["evolution_recente"] = np.where(
        data["moyenne_5"] > 0,
        ((data["moyenne_3"] - data["moyenne_5"]) / data["moyenne_5"]) * 100,
        0
    )

    # 8. Densité maritime (agrégée par année)
    density_par_annee = density.groupby("annee").agg(
        densite_moyenne=("vd", "mean"),
        densite_max=("vd", "max")
    ).reset_index()
    data = data.merge(density_par_annee, on="annee", how="left")
    data["densite_moyenne"] = data["densite_moyenne"].ffill().bfill()
    data["densite_max"] = data["densite_max"].ffill().bfill()

    # 9. Flotte
    fleet_total = fleet[fleet["type_navire"] == "Flotte totale"][["annee", "nombre_navires"]].drop_duplicates("annee")
    data = data.merge(fleet_total, on="annee", how="left")
    data["nombre_navires"] = data["nombre_navires"].ffill().bfill()

    # 10. Features finales
    FEATURES = [
        "annee", "densite_moyenne", "densite_max", "nombre_navires",
        "nb_accidents_lag_1", "nb_accidents_lag_2", "nb_accidents_lag_3",
        "moyenne_3", "moyenne_5", "evolution_recente"
    ]
    classes_ordre = ["Forte diminution", "Faible diminution", "Stable", "Faible augmentation", "Forte augmentation"]
    data["categorie_num"] = data["categorie"].map({classe: i for i, classe in enumerate(classes_ordre)})

    # 11. Suppression des lignes incomplètes
    data_modele = data.dropna(subset=FEATURES + ["categorie_num"]).copy()
    print(f"✅ {len(data_modele)} lignes utilisables pour l'entraînement")

    # 12. Train/Test temporel (2011-2022 = train, 2023-2025 = test)
    train = data_modele[data_modele["annee"] <= 2022]
    test = data_modele[data_modele["annee"].between(2023, 2025)]
    X_train, y_train = train[FEATURES], train["categorie_num"].astype(int)
    X_test, y_test = test[FEATURES], test["categorie_num"].astype(int)

    print(f"\n📊 Split temporel : {len(train)} train, {len(test)} test")

    # 13. Entraînement
    print("\n🌲 Entraînement du Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 14. Évaluation
    print("\n" + "=" * 70)
    print("ÉVALUATION 2023-2025")
    print("=" * 70)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy : {accuracy:.3f} ({accuracy * 100:.1f} %)")

    print("\n📋 Rapport de classification :")
    print(classification_report(
        y_test, y_pred,
        labels=list(range(len(classes_ordre))),
        target_names=classes_ordre,
        zero_division=0
    ))

    # 15. Matrice de confusion
    matrice = confusion_matrix(y_test, y_pred, labels=list(range(len(classes_ordre))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrice, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes_ordre, yticklabels=classes_ordre
    )
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion - Prédiction de tendance")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "confusion_matrix_tendance.png", dpi=300)
    plt.close()
    print(f"\n📸 Matrice de confusion sauvegardée dans {OUTPUT_PATH}")

    # 16. Importance des variables
    importance = pd.DataFrame({
        "Parametre": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\n📈 Importance des variables :")
    for _, row in importance.iterrows():
        print(f"  {row['Parametre']:25s} {row['Importance']:.3f}")

    # 17. Prédiction 2023-2030
    print("\n" + "=" * 70)
    print("PRÉDICTION DES TENDANCES 2023-2030")
    print("=" * 70)

    resultats = []
    future_years = list(range(2023, 2031))
    historique = data[[
        "zone", "annee", "nb_accidents", "densite_moyenne", "densite_max", "nombre_navires"
    ]].copy()

    def construire_features(historique_zone, annee):
        historique_zone = historique_zone.sort_values("annee").copy()
        accidents = historique_zone["nb_accidents"].tolist()
        if len(accidents) < 5:
            return None

        lag1, lag2, lag3 = accidents[-1], accidents[-2], accidents[-3]
        moyenne_3 = np.mean(accidents[-3:])
        moyenne_5 = np.mean(accidents[-5:])
        evolution_recente = ((moyenne_3 - moyenne_5) / moyenne_5 * 100) if moyenne_5 > 0 else 0

        return pd.DataFrame([{
            "annee": annee,
            "densite_moyenne": historique_zone["densite_moyenne"].dropna().iloc[-1],
            "densite_max": historique_zone["densite_max"].dropna().iloc[-1],
            "nombre_navires": historique_zone["nombre_navires"].dropna().iloc[-1],
            "nb_accidents_lag_1": lag1,
            "nb_accidents_lag_2": lag2,
            "nb_accidents_lag_3": lag3,
            "moyenne_3": moyenne_3,
            "moyenne_5": moyenne_5,
            "evolution_recente": evolution_recente
        }])

    # Prédictions
    for zone in zones:
        zone_data = historique[historique["zone"] == zone].sort_values("annee").copy()
        donnees_zone = zone_data.copy()

        for annee in future_years:
            X_future = construire_features(donnees_zone, annee)
            if X_future is None:
                continue

            prediction = model.predict(X_future[FEATURES])[0]
            probabilites = model.predict_proba(X_future[FEATURES])[0]
            tendance = classes_ordre[int(prediction)]
            confiance = probabilites[int(prediction)] * 100

            resultats.append({
                "zone": zone,
                "annee": annee,
                "tendance": tendance,
                "classe": int(prediction),
                "confiance": round(confiance, 1)
            })

            # Mise à jour de l'historique pour les années suivantes
            if annee <= 2025:
                nb = float(data[(data["zone"] == zone) & (data["annee"] == annee)]["nb_accidents"].iloc[0] if len(data[(data["zone"] == zone) & (data["annee"] == annee)]) > 0 else 0)
            else:
                nb = float(donnees_zone["nb_accidents"].iloc[-1])

            nouvelle_ligne = pd.DataFrame([{
                "zone": zone, "annee": annee, "nb_accidents": nb,
                "densite_moyenne": donnees_zone["densite_moyenne"].iloc[-1],
                "densite_max": donnees_zone["densite_max"].iloc[-1],
                "nombre_navires": donnees_zone["nombre_navires"].iloc[-1]
            }])
            donnees_zone = pd.concat([donnees_zone, nouvelle_ligne], ignore_index=True)

    df_futur = pd.DataFrame(resultats)

    # 18. Distribution des tendances
    print("\n📊 Distribution des tendances prédites :")
    for annee in future_years:
        print(f"\n  {annee} :")
        sous_df = df_futur[df_futur["annee"] == annee]
        for classe in classes_ordre:
            nombre = len(sous_df[sous_df["tendance"] == classe])
            print(f"    {classe:25s} : {nombre} zones")

    # 19. Sauvegarde
    output_file = DATA_PATH / "predictions_tendance_2023_2030.csv"
    os.makedirs(DATA_PATH, exist_ok=True)
    df_futur.to_csv(output_file, index=False)
    print(f"\n✅ Prédictions sauvegardées : {output_file}")

    return df_futur

# ============================================================
# EXÉCUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LANCEMENT : Modèle de classification des tendances")
    print("=" * 70 + "\n")
    df_tendance = run_classification_tendance()