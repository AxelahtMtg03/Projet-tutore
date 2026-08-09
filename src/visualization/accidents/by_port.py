import os

import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_processing.loader import charger_donnees_finales

def accidents_par_annee_port(port:str):
    total = charger_donnees_finales()
    total = total[total["port"] == port]
    comptage = total.groupby("annee").size()
    return comptage

def graphique_port_annee(total,port):
    plt.figure(figsize=(12,6))

    plt.plot(total.index,total.values,marker="o"
    )

    plt.title(f"Nombre d'accidents par année dans la zone : {port}", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True)

    os.makedirs("visualization/accidents", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"visualization/accidents/graph_accident_by_port_{port.replace(' ', '_')}.png", dpi=150)
    plt.close()
# comptage_port = port_global()
# print(comptage_port)
# top10 = comptage_port.head(100)
# print("\n=== TOP 100 DES PORTS LES PLUS ACCIDENTOGÈNES ===")
# for i, (port, count) in enumerate(top10.items(), 1):
#     print(f"{i}. {port}: {count} accidents")

a="GERMANY - Hamburg"
graphique_port_annee(accidents_par_annee_port(a),a)