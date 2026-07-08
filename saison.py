import matplotlib.pyplot as plt
from charge import charger_donnees 


def saison_global():
    # Fusionner tous les fichiers
    total = charger_donnees()
    comptage_total = total['saison'].value_counts()
    ordre_saisons = ['Printemps', 'Été', 'Automne', 'Hiver']
    comptage_total = comptage_total.reindex(ordre_saisons)
    return comptage_total

def grapique_saison_global(total):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(total.index, total.values, 
                color=['green', 'red', 'orange', 'blue'])
    plt.title('Nombre d\'accidents par saison', fontsize=14)
    plt.xlabel('Saison', fontsize=12)
    plt.ylabel('Nombre d\'accidents', fontsize=12)
    plt.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()
    
def accidents_par_annee_saison(saison:str):
    total = charger_donnees()
    total = total[total["saison"] == saison]
    comptage = total.groupby("annee").size()
    return comptage

def graphique_saison_annee(total,saison):
    plt.figure(figsize=(12,6))

    plt.plot(total.index,total.values,marker="o"
    )

    plt.title(f"Nombre d'accidents en {saison} par année", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True)

    plt.show()

def accidents_par_annee_saison():
    total = charger_donnees()
    comptage = total.groupby(["annee","saison"]).size().unstack(fill_value=0)
    return comptage
def graphique_saison_annee(comptage):

    plt.figure(figsize=(12,6))

    for saison in comptage.columns:
        plt.plot(
            comptage.index,
            comptage[saison],
            marker="o",
            label=saison
        )

    plt.title("Nombre d'accidents par saison et par année")
    plt.xlabel("Année")
    plt.ylabel("Nombre d'accidents")

    plt.grid(True)
    plt.legend()

    plt.show()
    
# grapique_saison_global(saison_global())
# saison = "Été"
# graphique_saison_annee(accidents_par_annee_saison(saison),saison)
graphique_saison_annee(accidents_par_annee_saison())
