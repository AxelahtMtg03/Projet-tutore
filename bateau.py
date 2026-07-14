import matplotlib.pyplot as plt
from charge import *


def type_bateau_global():
    df_total = charger_donnees_finales()
    comptage_total = df_total['bateau'].value_counts() #Compte nb accidents pour chaque type de bateau + trie du plus au moins frequent
    return comptage_total

def graphique_type_bateau(total):
    plt.figure(figsize=(14, 6)) 
    bars = plt.bar(total.index, total.values, color='blue')
    plt.title('Nombre d accident par type de bateau', fontsize=14)
    plt.xlabel('Type de bateau', fontsize=12)
    plt.ylabel('Nombre d accident', fontsize=12)
    plt.xticks(rotation=45, ha='right') #Evite que lignes se superposent
    plt.grid(True, alpha=0.3) #Ajoute grille en arrière plan

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.show()

def accidents_par_annee_bateau(bateau:str):
    total = charger_donnees_finales()
    total = total[total["bateau"] == bateau]
    comptage = total.groupby("annee").size()
    return comptage

def graphique_bateau_annee(total,bateau):
    plt.figure(figsize=(12,6))

    plt.plot(total.index,total.values,marker="o"
    )

    plt.title(f"Nombre d'accidents des {bateau} par année", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True)

    plt.show()

#graphique_type_bateau(type_bateau_global())
a="Submersible"
graphique_bateau_annee(accidents_par_annee_bateau(a),a)