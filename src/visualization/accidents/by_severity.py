import os

import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_processing.loader import charger_donnees_finales

def gravite_global():
    # Fusionner tous les fichiers
    total = charger_donnees_finales()
    total_europe = total[
        (total["lat"].between(35, 70)) &   # Latitude Europe
        (total["long"].between(-10, 40))   # Longitude Europe
    ]
    comptage = total_europe['gravite'].value_counts().sort_index()
    comptage = comptage[comptage.index != 'nan']
    return comptage

def graphique_gravite(total):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(total.index, total.values)
  
    plt.title('Nombre d\'accidents en fonction de la gravité de l\'accident', fontsize=14)
    plt.xlabel('Gravité de l\'accident', fontsize=12)
    plt.ylabel('Nombre d\'accidents', fontsize=12)
    plt.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    os.makedirs("visualization/accidents", exist_ok=True)
    plt.tight_layout()
    plt.savefig("visualization/accidents/graph_accident_by_severity_europe.png", dpi=150)
    plt.close()

graphique_gravite(gravite_global())