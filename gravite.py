import matplotlib.pyplot as plt
from charge import charger_donnees

def gravite_global():
    # Fusionner tous les fichiers
    total = charger_donnees()
    comptage = total['gravite'].value_counts().sort_index()
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

    plt.tight_layout()
    plt.show()

graphique_gravite(gravite_global())