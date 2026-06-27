import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

dossier = 'données_d_accident_en_mer/'
fichiers = glob.glob(os.path.join(dossier, '*.csv'))

def accidents_par_annee():
    total = pd.DataFrame()
    for f in fichiers:
        df = pd.read_csv(f)
        
        df['Date of occurrence'] = pd.to_datetime(df['Date of occurrence'], errors='coerce')
        df = df.dropna(subset=["Date of occurrence"])

        df['annee'] = df['Date of occurrence'].dt.year
        
        total = pd.concat([total, df], ignore_index=True)


    comptage = total.groupby("annee").size()

    return comptage
def graphique_annee(total):
    plt.figure(figsize=(12,6))

    plt.plot(total.index,total.values,marker="o"
    )

    plt.title(f"Nombre d'accidents par année", fontsize=14)
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True)

    plt.show()

graphique_annee(accidents_par_annee())