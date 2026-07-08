import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import re
dossier = 'données_d_accident_en_mer/'
fichiers = glob.glob(os.path.join(dossier, '*.csv'))

def get_time(heure:float)->str:
    """on renvoie juste l'heure sans prendre en compte les minutes afin d'obtenir des plages horaires plus simple"""
    if pd.isna(heure) or heure == "":
        return None
    return str(heure).split(":")[0]

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

def charger_donnees():
    total = pd.DataFrame()

    for f in fichiers:
        df = pd.read_csv(f)
        df['gravite'] = df['Occurrence severity'] #gravite de l'accident
        df['long'] = df['Longitude'] #longitude
        df['lat'] = df['Latitude'] #latitude
        df['heure'] = df['Time (LT) of occurrence'].apply(get_time) #le temps en heure
        df = df.dropna(subset=["heure"])
        df['port'] = df['Port of accident'] #lieu de l'accident (port)
        df['Date of occurrence'] = pd.to_datetime(df['Date of occurrence'], errors='coerce') #date de l'accident
        df = df.dropna(subset=["Date of occurrence"])
        df['saison'] = df['Date of occurrence'].apply(get_saison) #saison ou on lieux les accidents
        df['annee'] = df['Date of occurrence'].dt.year #année
        df['bateau'] = df['Ship / craft type'].str.split(' - ').str[0] #type de bateau
        df = df.dropna(subset=["bateau"])
        df = df[df["bateau"] != "Unknown"]

        total = pd.concat([total, df], ignore_index=True)
    print(f"Total des enregistrements: {len(total)}")
    return total


def compatage_croise(para1:str,para2:str):
    total = charger_donnees()
    comptage = total.groupby([para1,para2]).size().unstack(fill_value=0)
    return comptage

def graphique(comptage,para1:str,para2:str):

    plt.figure(figsize=(12,6))

    for x in comptage.columns:
        plt.plot(
            comptage.index,
            comptage[x],
            marker="o",
            label=x
        )

    plt.title("Nombre d'accidents par " + para1 +" et par " +para2)
    plt.xlabel(para2)
    plt.ylabel("Nombre d'accidents")

    plt.grid(True)
    plt.legend()

    plt.show()
b="heure"
a="bateau"
# graphique((compatage_croise(b,a)),b,a)
# df = charger_donnees()
# df.to_csv("accidents_nettoyes.csv", index=False)

def charger_donnees_sans_doublons():
    df = charger_donnees()  # Utilise ta fonction existante
    
    # Colonnes à considérer pour les doublons
    colonnes_doublons = [
        'Date of occurrence',
        'Ship / craft type', 
        'Latitude',
        'Longitude',
        'Time (LT) of occurrence',
        'Port of accident'
    ]
    
    # Supprime les doublons
    df_sans_doublons = df.drop_duplicates(subset=colonnes_doublons, keep='first')
    
    print(f"\nDoublons supprimés : {len(df) - len(df_sans_doublons)}")
    print(f"Enregistrements finaux : {len(df_sans_doublons)}")
    
    return df_sans_doublons

# Utilisation :
df_final = charger_donnees_sans_doublons()
df_final.to_csv("accidents_final_nettoyes.csv", index=False)