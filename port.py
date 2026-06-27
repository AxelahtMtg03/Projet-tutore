import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

dossier = 'données_d_accident_en_mer/'
fichiers = glob.glob(os.path.join(dossier, '*.csv'))

def get_port(port:str)->str:
    """on renvoie juste l'heure sans prendre en compte les minutes afin d'obtenir des plages horaires plus simple"""
    if port=='' or port==None:
        return None
    return port
    
def port_global():
    # Fusionner tous les fichiers
    total = pd.DataFrame()

    for f in fichiers:
        df = pd.read_csv(f)
        # Appliquer l'heure
        df['port'] = df['Port of accident'].apply(get_port)
        total = pd.concat([total, df], ignore_index=True)

    print(f"Total des enregistrements: {len(total)}")

    comptage_port = total['port'].value_counts().sort_values(ascending=False)
    comptage_port = comptage_port[comptage_port.index != 'nan']
    return comptage_port

def graphique_port(comptage_port):
    plt.figure(figsize=(14, 6))
    bars = plt.bar(comptage_port.index, comptage_port.values, 
                color='steelblue', edgecolor='black', alpha=0.8)

    plt.title('Nombre d\'accidents par port', fontsize=16, fontweight='bold')
    plt.xlabel('Port dif', fontsize=12)
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
comptage_port = port_global()
top10 = comptage_port.head(100)
print("\n=== TOP 100 DES PORTS LES PLUS ACCIDENTOGÈNES ===")
for i, (port, count) in enumerate(top10.items(), 1):
    print(f"{i}. {port}: {count} accidents")
