# evaluation_modele.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

print("=" * 60)
print("📊 ÉVALUATION DU MODÈLE DE TENDANCE")
print("=" * 60)

# ============================================================
# 1. CHARGER LES PRÉDICTIONS
# ============================================================

print("\n1. Chargement des données...")

# Charger les prédictions
df_pred = pd.read_csv("data/processed/predictions_tendance_2023_2030.csv")

# Compter les lignes
print(f"   {len(df_pred)} prédictions chargées")
print(f"   Colonnes: {df_pred.columns.tolist()}")

# ============================================================
# 2. STATISTIQUES GLOBALES
# ============================================================

print("\n" + "=" * 60)
print("📊 STATISTIQUES GLOBALES")
print("=" * 60)

# Distribution des tendances par année
print("\n📈 Distribution des tendances par année:")
for annee in sorted(df_pred['annee'].unique()):
    df_annee = df_pred[df_pred['annee'] == annee]
    print(f"\n   {int(annee)}:")
    for tendance in df_annee['tendance'].value_counts().index:
        count = len(df_annee[df_annee['tendance'] == tendance])
        pct = (count / len(df_annee)) * 100
        print(f"      {tendance}: {count} zones ({pct:.1f}%)")

# ============================================================
# 3. STATISTIQUES DE CONFIANCE
# ============================================================

print("\n" + "=" * 60)
print("📊 STATISTIQUES DE CONFIANCE")
print("=" * 60)

# Si la colonne confiance existe
if 'confiance' in df_pred.columns:
    print(f"\n   Confiance moyenne: {df_pred['confiance'].mean():.1f}%")
    print(f"   Confiance médiane: {df_pred['confiance'].median():.1f}%")
    print(f"   Confiance min: {df_pred['confiance'].min():.1f}%")
    print(f"   Confiance max: {df_pred['confiance'].max():.1f}%")

# Distribution de la confiance par tendance
if 'confiance' in df_pred.columns:
    print("\n📊 Confiance par tendance:")
    for tendance in df_pred['tendance'].unique():
        df_tendance = df_pred[df_pred['tendance'] == tendance]
        print(f"   {tendance}: {df_tendance['confiance'].mean():.1f}%")

# ============================================================
# 4. GRAPHIQUE 1 : ÉVOLUTION DES TENDANCES
# ============================================================

print("\n📈 Génération des graphiques...")

# Créer un tableau croisé
tendance_evolution = pd.crosstab(df_pred['annee'], df_pred['tendance'])

# Graphique
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Graphique 1 : Évolution des tendances
ax1 = axes[0, 0]
for tendance in tendance_evolution.columns:
    ax1.plot(tendance_evolution.index, tendance_evolution[tendance], 
             marker='o', label=tendance, linewidth=2)
ax1.set_xlabel('Année')
ax1.set_ylabel("Nombre de zones")
ax1.set_title('Évolution des tendances (2023-2030)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graphique 2 : Distribution des tendances (2025)
ax2 = axes[0, 1]
df_2025 = df_pred[df_pred['annee'] == 2025]
counts = df_2025['tendance'].value_counts()
colors = ['#08306b', '#27ae60', '#8e44ad', '#f4d03f', '#e74c3c']
bars = ax2.bar(counts.index, counts.values, color=colors[:len(counts)])
ax2.set_xlabel('Tendance')
ax2.set_ylabel("Nombre de zones")
ax2.set_title('Distribution des tendances en 2025')
ax2.tick_params(axis='x', rotation=15)
ax2.grid(True, alpha=0.3)

# Graphique 3 : Confiance par tendance (si disponible)
ax3 = axes[1, 0]
if 'confiance' in df_pred.columns:
    confiance_moyenne = df_pred.groupby('tendance')['confiance'].mean()
    bars = ax3.bar(confiance_moyenne.index, confiance_moyenne.values, 
                   color=colors[:len(confiance_moyenne)])
    ax3.set_xlabel('Tendance')
    ax3.set_ylabel('Confiance moyenne (%)')
    ax3.set_title('Confiance moyenne par tendance')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(True, alpha=0.3)

# Graphique 4 : Évolution de la confiance (si disponible)
ax4 = axes[1, 1]
if 'confiance' in df_pred.columns:
    confiance_par_annee = df_pred.groupby('annee')['confiance'].mean()
    ax4.plot(confiance_par_annee.index, confiance_par_annee.values, 
             marker='o', color='purple', linewidth=2, markersize=8)
    ax4.set_xlabel('Année')
    ax4.set_ylabel('Confiance moyenne (%)')
    ax4.set_title('Évolution de la confiance')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("visualization/prediction/evaluation_tendances.png", dpi=300)
print(f"✅ Graphique sauvegardé: visualization/predictions/evaluation_tendances.png")
plt.show()

# ============================================================
# 5. INTERPRÉTATION
# ============================================================

print("\n" + "=" * 60)
print("📝 INTERPRÉTATION")
print("=" * 60)

# Trouver la tendance dominante en 2030
df_2030 = df_pred[df_pred['annee'] == 2030]
tendance_dominante = df_2030['tendance'].value_counts().index[0]
pct_dominante = (df_2030['tendance'].value_counts().iloc[0] / len(df_2030)) * 100

print(f"\n✅ Tendance dominante en 2030: {tendance_dominante} ({pct_dominante:.1f}%)")

# Évolution de la tendance dominante
for annee in sorted(df_pred['annee'].unique()):
    df_annee = df_pred[df_pred['annee'] == annee]
    dominante = df_annee['tendance'].value_counts().index[0]
    pct = (df_annee['tendance'].value_counts().iloc[0] / len(df_annee)) * 100
    print(f"   {int(annee)}: {dominante} ({pct:.1f}%)")

# Confiance globale
if 'confiance' in df_pred.columns:
    confiance_moyenne = df_pred['confiance'].mean()
    if confiance_moyenne > 70:
        print(f"\n✅ Modèle TRÈS CONFANT: confiance moyenne = {confiance_moyenne:.1f}%")
    elif confiance_moyenne > 50:
        print(f"\n✅ Modèle PLUTÔT CONFANT: confiance moyenne = {confiance_moyenne:.1f}%")
    else:
        print(f"\n⚠️ Modèle INCERTAIN: confiance moyenne = {confiance_moyenne:.1f}%")
        print("   → Peut-être besoin de plus de données ou de meilleures variables")

print("\n" + "=" * 60)
print("✅ ÉVALUATION TERMINÉE")
print("=" * 60)