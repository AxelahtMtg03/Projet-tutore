import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

dossier = 'données_d_accident_en_mer/'
fichiers = glob.glob(os.path.join(dossier, '*.csv'))

def get_saison(date):
    m, d = date.month, date.day
    if (m == 3 and d >= 20) or (3 < m < 6) or (m == 6 and d < 21):
        return 'Printemps'
    elif (m == 6 and d >= 21) or (6 < m < 9) or (m == 9 and d < 22):
        return 'Été'
    elif (m == 9 and d >= 22) or (9 < m < 12) or (m == 12 and d < 21):
        return 'Automne'
    else:
        return 'Hiver'

def saison_global():
    # Fusionner tous les fichiers
    total = pd.DataFrame()
    for f in fichiers:
        df = pd.read_csv(f)
        
        # Convertir en datetime (avec gestion d'erreur)
        df['Date of occurrence'] = pd.to_datetime(df['Date of occurrence'], errors='coerce')
            
        # Appliquer la saison
        df['saison'] = df['Date of occurrence'].apply(get_saison)
        
        total = pd.concat([total, df], ignore_index=True)
        

    print(f"Total: {len(total)} enregistrements")

    # Compter par saison (tous fichiers confondus)
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
    total = pd.DataFrame()
    for f in fichiers:
        df = pd.read_csv(f)
        
        df['Date of occurrence'] = pd.to_datetime(df['Date of occurrence'], errors='coerce')
        df = df.dropna(subset=["Date of occurrence"])

        df['saison'] = df['Date of occurrence'].apply(get_saison)
        df['annee'] = df['Date of occurrence'].dt.year
        
        total = pd.concat([total, df], ignore_index=True)

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

# grapique_saison_global(saison_global())
saison = "Été"
graphique_saison_annee(accidents_par_annee_saison(saison),saison)