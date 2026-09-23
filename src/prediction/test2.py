"""
Analyse le fichier grid_search_pivot.csv et affiche :
  - la meilleure combinaison par score d'equilibre
  - la meilleure combinaison par accuracy globale
  - la meilleure combinaison par accuracy sur Augmentation
  - la meilleure combinaison par accuracy sur Diminution
"""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
FICHIER_PIVOT = DATA_PATH / "grid_search_pivot.csv"

if not FICHIER_PIVOT.exists():
    print(f"Fichier introuvable : {FICHIER_PIVOT}")
    raise SystemExit

df = pd.read_csv(FICHIER_PIVOT, encoding="utf-8-sig")
print(f"Lignes : {len(df)}")
print(f"Colonnes : {list(df.columns)}")
print()

# ------------------------------------------------------------
# 1. Meilleur score d'equilibre
# ------------------------------------------------------------
if "score_equilibre" in df.columns:
    best_equilibre = df.sort_values("score_equilibre", ascending=False).iloc[0]
    print("=" * 70)
    print("MEILLEUR SCORE D'EQUILIBRE (min(Diminution, Augmentation))")
    print("=" * 70)
    print(best_equilibre.to_string())
    print()

# ------------------------------------------------------------
# 2. Meilleure accuracy globale
# ------------------------------------------------------------
if "Global" in df.columns:
    best_global = df.sort_values("Global", ascending=False).iloc[0]
    print("=" * 70)
    print("MEILLEURE ACCURACY GLOBALE")
    print("=" * 70)
    print(best_global.to_string())
    print()

# ------------------------------------------------------------
# 3. Meilleure accuracy sur Augmentation
# ------------------------------------------------------------
if "Augmentation" in df.columns:
    best_aug = df.sort_values("Augmentation", ascending=False).iloc[0]
    print("=" * 70)
    print("MEILLEURE ACCURACY SUR AUGMENTATION")
    print("=" * 70)
    print(best_aug.to_string())
    print()

# ------------------------------------------------------------
# 4. Meilleure accuracy sur Diminution
# ------------------------------------------------------------
if "Diminution" in df.columns:
    best_dim = df.sort_values("Diminution", ascending=False).iloc[0]
    print("=" * 70)
    print("MEILLEURE ACCURACY SUR DIMINUTION")
    print("=" * 70)
    print(best_dim.to_string())
    print()

# ------------------------------------------------------------
# 5. Top 10 equilibre
# ------------------------------------------------------------
print("=" * 70)
print("TOP 10 PAR SCORE D'EQUILIBRE")
print("=" * 70)
cols = [c for c in ["seuil_ecart", "seuil_pct", "granularite",
                    "Diminution", "Stable", "Augmentation", "Global",
                    "score_equilibre"] if c in df.columns]
print(df.sort_values("score_equilibre", ascending=False)[cols].head(10).to_string(index=False))
print()

# ------------------------------------------------------------
# 6. Top 10 Global
# ------------------------------------------------------------
print("=" * 70)
print("TOP 10 PAR ACCURACY GLOBALE")
print("=" * 70)
print(df.sort_values("Global", ascending=False)[cols].head(10).to_string(index=False))