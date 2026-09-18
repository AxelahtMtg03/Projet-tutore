# # src/prediction/teste.py
# import pandas as pd
# import numpy as np
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).parent.parent.parent
# DATA_PATH = PROJECT_ROOT / "data" / "processed"


# def analyser_predictions(fichier_csv, periode="mensuel"):
#     """
#     Analyse les predictions et explique le calcul de l'accuracy.
#     """
    
#     print("=" * 70)
#     print(f"ANALYSE DES PREDICTIONS - {periode.upper()}")
#     print("=" * 70)
    
#     # 1. Charger les predictions
#     try:
#         df_pred = pd.read_csv(DATA_PATH / fichier_csv)
#         print(f"\nFichier charge : {fichier_csv}")
#         print(f"Total de predictions : {len(df_pred)}")
#     except FileNotFoundError:
#         print(f"Fichier non trouve : {fichier_csv}")
#         return
    
#     # 2. Afficher les colonnes
#     print(f"\nColonnes : {df_pred.columns.tolist()}")
    
#     # 3. Repartition par annee
#     print("\n" + "-" * 70)
#     print("REPARTITION PAR ANNEE (toutes les predictions)")
#     print("-" * 70)
#     repartition_annee = df_pred.groupby("annee").size()
#     for annee, count in repartition_annee.items():
#         print(f"  {annee} : {count} predictions")
    
#     # 4. Filtrer 2023-2025 (annees comparees)
#     print("\n" + "-" * 70)
#     print("FILTRAGE 2023-2025 (annees comparees)")
#     print("-" * 70)
    
#     if "annee" in df_pred.columns:
#         df_pred_2023_2025 = df_pred[df_pred["annee"].between(2023, 2025)]
#         print(f"Predictions 2023-2025 : {len(df_pred_2023_2025)}")
#         print(f"Predictions 2026-2030 : {len(df_pred) - len(df_pred_2023_2025)}")
    
#     # 5. Repartition des tendances (CSV complet)
#     print("\n" + "-" * 70)
#     print("REPARTITION DES TENDANCES (CSV complet)")
#     print("-" * 70)
#     repartition_tendance = df_pred["tendance"].value_counts()
#     for tendance, count in repartition_tendance.items():
#         print(f"  {tendance} : {count}")
    
#     # 6. Repartition des tendances (2023-2025)
#     print("\n" + "-" * 70)
#     print("REPARTITION DES TENDANCES (2023-2025 seulement)")
#     print("-" * 70)
#     if "annee" in df_pred.columns:
#         repartition_tendance_2023_2025 = df_pred_2023_2025["tendance"].value_counts()
#         for tendance, count in repartition_tendance_2023_2025.items():
#             print(f"  {tendance} : {count}")
    
#     # 7. Comparaison avec la realite
#     print("\n" + "-" * 70)
#     print("COMPARAISON AVEC LA REALITE")
#     print("-" * 70)
    
#     # Charger les accidents reels
#     accidents = pd.read_csv(DATA_PATH / "maritime_accidents.csv")
#     accidents = accidents.dropna(subset=["lat", "long", "annee"])
    
#     def assigner_zone(row):
#         lat_zone = np.floor(row["lat"] / 1.5) * 1.5
#         lon_zone = np.floor(row["long"] / 1.5) * 1.5
#         return f"z_{lat_zone:.1f}_{lon_zone:.1f}"
    
#     accidents["zone"] = accidents.apply(assigner_zone, axis=1)
    
#     # Selon la periode
#     if periode == "mensuel":
#         accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"]).dt.month.astype(int)
#         accidents_par_zone = accidents.groupby(["zone", "annee", "mois"]).size().reset_index(name="nb_accidents")
#         ref_zone = accidents_par_zone[accidents_par_zone["annee"] <= 2022].groupby(["zone", "mois"])["nb_accidents"].median().reset_index()
#         ref_zone.columns = ["zone", "mois", "reference_historique"]
#         data = accidents_par_zone.merge(ref_zone, on=["zone", "mois"], how="left")
    
#     elif periode == "annuel":
#         accidents_par_zone = accidents.groupby(["zone", "annee"]).size().reset_index(name="nb_accidents")
#         ref_zone = accidents_par_zone[accidents_par_zone["annee"] <= 2022].groupby("zone")["nb_accidents"].median().reset_index()
#         ref_zone.columns = ["zone", "reference_historique"]
#         data = accidents_par_zone.merge(ref_zone, on="zone", how="left")
    
#     elif periode == "semestriel":
#         accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"]).dt.month
#         accidents["semestre"] = np.where(accidents["mois"] <= 6, 1, 2)
#         accidents_par_zone = accidents.groupby(["zone", "annee", "semestre"]).size().reset_index(name="nb_accidents")
#         ref_zone = accidents_par_zone[accidents_par_zone["annee"] <= 2022].groupby(["zone", "semestre"])["nb_accidents"].median().reset_index()
#         ref_zone.columns = ["zone", "semestre", "reference_historique"]
#         data = accidents_par_zone.merge(ref_zone, on=["zone", "semestre"], how="left")
    
#     elif periode == "trimestriel":
#         accidents["mois"] = pd.to_datetime(accidents["Date of occurrence"]).dt.month
#         accidents["trimestre"] = np.ceil(accidents["mois"] / 3).astype(int)
#         accidents_par_zone = accidents.groupby(["zone", "annee", "trimestre"]).size().reset_index(name="nb_accidents")
#         ref_zone = accidents_par_zone[accidents_par_zone["annee"] <= 2022].groupby(["zone", "trimestre"])["nb_accidents"].median().reset_index()
#         ref_zone.columns = ["zone", "trimestre", "reference_historique"]
#         data = accidents_par_zone.merge(ref_zone, on=["zone", "trimestre"], how="left")
    
#     data["reference_historique"] = data["reference_historique"].fillna(0)
    
#     def categoriser(row):
#         if row["reference_historique"] < 0.5:
#             return "Donnees insuffisantes"
#         evolution = (row["nb_accidents"] - row["reference_historique"]) / row["reference_historique"] * 100
#         ecart_absolu = row["nb_accidents"] - row["reference_historique"]
#         if abs(ecart_absolu) < 1.5:
#             return "Stable"
#         elif evolution < -30:
#             return "Forte diminution"
#         elif evolution < -10:
#             return "Faible diminution"
#         elif evolution < 10:
#             return "Stable"
#         elif evolution < 30:
#             return "Faible augmentation"
#         else:
#             return "Forte augmentation"
    
#     data["tendance_reelle"] = data.apply(categoriser, axis=1)
#     data = data[data["annee"].between(2023, 2025)]
    
#     # Fusionner
#     if periode == "mensuel":
#         df_pred_eval = df_pred.copy()
#         if df_pred_eval["mois"].dtype == "object":
#             df_pred_eval["mois"] = df_pred_eval["mois"].str.split('-').str[1].astype(int)
#         else:
#             df_pred_eval["mois"] = df_pred_eval["mois"].astype(int)
#         df_compare = df_pred_eval.merge(data[["zone", "annee", "mois", "tendance_reelle"]], on=["zone", "annee", "mois"], how="inner")
    
#     elif periode == "annuel":
#         df_compare = df_pred.merge(data[["zone", "annee", "tendance_reelle"]], on=["zone", "annee"], how="inner")
    
#     elif periode == "semestriel":
#         df_pred["semestre"] = df_pred["semestre"].astype(int)
#         data["semestre"] = data["semestre"].astype(int)
#         df_compare = df_pred.merge(data[["zone", "annee", "semestre", "tendance_reelle"]], on=["zone", "annee", "semestre"], how="inner")
    
#     elif periode == "trimestriel":
#         df_pred["trimestre"] = df_pred["trimestre"].astype(int)
#         data["trimestre"] = data["trimestre"].astype(int)
#         df_compare = df_pred.merge(data[["zone", "annee", "trimestre", "tendance_reelle"]], on=["zone", "annee", "trimestre"], how="inner")
    
#     df_compare = df_compare[df_compare["tendance_reelle"] != "Donnees insuffisantes"]
    
#     print(f"Zones comparees (2023-2025) : {len(df_compare)}")
    
#     # 8. Accuracy par tendance
#     print("\n" + "-" * 70)
#     print("ACCURACY PAR TENDANCE (sur 2023-2025)")
#     print("-" * 70)
    
#     df_compare["correct"] = df_compare["tendance"] == df_compare["tendance_reelle"]
    
#     for classe in ["Forte diminution", "Faible diminution", "Stable", "Faible augmentation", "Forte augmentation"]:
#         df_classe = df_compare[df_compare["tendance_reelle"] == classe]
#         if len(df_classe) > 0:
#             nb_correct = df_classe["correct"].sum()
#             acc = df_classe["correct"].mean() * 100
#             print(f"\n  {classe} :")
#             print(f"    - Zones avec tendance reelle '{classe}' : {len(df_classe)}")
#             print(f"    - Bonnes predictions : {nb_correct}")
#             print(f"    - Accuracy : {nb_correct}/{len(df_classe)} = {acc:.1f}%")
    
#     # 9. Accuracy globale
#     accuracy = df_compare["correct"].mean() * 100
#     print("\n" + "=" * 70)
#     print(f"ACCURACY GLOBALE : {accuracy:.1f}%")
#     print("=" * 70)
    
#     # 10. Explication du filtrage
#     print("\n" + "-" * 70)
#     print("EXPLICATION DU FILTRAGE")
#     print("-" * 70)
#     print(f"  CSV complet : {len(df_pred)} predictions")
#     print(f"  Apres filtre 2023-2025 : {len(df_pred[df_pred['annee'].between(2023, 2025)])} predictions")
#     print(f"  Apres fusion avec la realite : {len(df_compare) + len(df_compare[df_compare['tendance_reelle'] == 'Donnees insuffisantes'])} predictions")
#     print(f"  Apres filtre 'Donnees insuffisantes' : {len(df_compare)} predictions")
    
#     return df_compare


# if __name__ == "__main__":
#     # Analyser le mensuel
#     analyser_predictions("predictions_tendance_mensuelle_2023_2030.csv", periode="mensuel")
    
#     # Analyser l'annuel
#     print("\n\n")
#     analyser_predictions("predictions_tendance_cv_2023_2030.csv", periode="annuel")
    
#     # Analyser le semestriel
#     print("\n\n")
#     analyser_predictions("predictions_tendance_semestrielle_2023_2030.csv", periode="semestriel")
    
#     # Analyser le trimestriel
#     print("\n\n")
#     analyser_predictions("predictions_tendance_trimestrielle_2023_2030.csv", periode="trimestriel")
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed")

# 1. CSV complet
df_pred = pd.read_csv(DATA_PATH / "predictions_tendance_mensuelle_2023_2030.csv")
print("=== CSV COMPLET ===")
print(df_pred["tendance"].value_counts())
print(f"\nTotal : {len(df_pred)}")

# 2. CSV 2023-2025
df_pred_2023_2025 = df_pred[df_pred["annee"].between(2023, 2025)]
print("\n=== CSV 2023-2025 ===")
print(df_pred_2023_2025["tendance"].value_counts())
print(f"\nTotal : {len(df_pred_2023_2025)}")

# 3. Réalité
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
data_2023_2025 = data[data["annee"].between(2023, 2025)]
print("\n=== RÉALITÉ 2023-2025 ===")
print(data_2023_2025["tendance_reelle"].value_counts())
print(f"\nTotal : {len(data_2023_2025)}")
# Zones dans le CSV
zones_csv = set(df_pred_2023_2025["zone"].unique())
print(f"Zones dans le CSV : {len(zones_csv)}")

# Zones dans la réalité
zones_reel = set(data_2023_2025["zone"].unique())
print(f"Zones dans la réalité : {len(zones_reel)}")

# Intersection
zones_communes = zones_csv & zones_reel
print(f"Zones communes : {len(zones_communes)}")

# Zones dans le CSV mais pas dans la réalité
zones_csv_seul = zones_csv - zones_reel
print(f"Zones dans le CSV mais pas dans la réalité : {len(zones_csv_seul)}")