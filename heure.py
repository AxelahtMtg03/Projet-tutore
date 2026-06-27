import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

dossier = 'données_d_accident_en_mer/'
fichiers = glob.glob(os.path.join(dossier, '*.csv'))

def get_time(heure:float)->str:
    """on renvoie juste l'heure sans prendre en compte les minutes afin d'obtenir des plages horaires plus simple"""
    if heure=='':
        return None
    return str(heure).split(":")[0]
    
def heure_global():
    # Fusionner tous les fichiers
    df_total = pd.DataFrame()

    for f in fichiers:
        df = pd.read_csv(f)
        # Appliquer l'heure
        df['heure'] = df['Time (LT) of occurrence'].apply(get_time)
        df_total = pd.concat([df_total, df], ignore_index=True)

    print(f"Total des enregistrements: {len(df_total)}")

    comptage_heures = df_total['heure'].value_counts().sort_index()
    comptage_heures = comptage_heures[comptage_heures.index != 'nan']
    return comptage_heures

def graphique_heure(comptage_heures):
    plt.figure(figsize=(14, 6))
    bars = plt.bar(comptage_heures.index, comptage_heures.values, 
                color='steelblue', edgecolor='black', alpha=0.8)

    plt.title('Nombre d\'accidents par heure', fontsize=16, fontweight='bold')
    plt.xlabel('Heure de la journée', fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(range(0, 24, 1))

    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

graphique_heure(heure_global())