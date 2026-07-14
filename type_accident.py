import matplotlib.pyplot as plt
from charge import *
import textwrap


def type_accident_global():
    """Compte le nombre d'accidents pour chaque type d'accident, trié du plus au moins fréquent"""
    total = charger_donnees_finales()
    comptage = total['type_accident'].value_counts()
    return comptage

def graphique_type_accident(total):
    plt.figure(figsize=(14, 6))
    bars = plt.bar(total.index, total.values, color='blue')
    plt.title("Nombre d'accidents par type d'accident", fontsize=14)
    plt.xlabel("Type d'accident", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.xticks(rotation=45, ha='right') 
    plt.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.show()

def accidents_par_annee_pour_type(type_accident:str):
    """le nombre type d'accident par année"""
    total = charger_donnees_finales()
    total = total[total["type_accident"] == type_accident]
    comptage = total.groupby("annee").size()
    return comptage

def accidents_par_annee_type():
    """Compte le nombre d'accidents par année, pour chaque type d'accident"""
    total = charger_donnees_finales()
    comptage = total.groupby(["annee", "type_accident"]).size().unstack(fill_value=0)
    return comptage

def graphique_type_annee(comptage):
    """Affiche, pour chaque type d'accident, l'évolution du nombre d'accidents par année"""
    plt.figure(figsize=(12, 6))

    for type_accident in comptage.columns:
        plt.plot(
            comptage.index,
            comptage[type_accident],
            marker="o",
            label=type_accident
        )

    plt.title("Nombre d'accidents par type d'accident et par année")
    plt.xlabel("Année")
    plt.ylabel("Nombre d'accidents")
    plt.grid(True)
    plt.legend()
    plt.show()

def cause_accident_humain_global():
    """Compte le nombre d'accidents pour chaque cause d'accident humain, triée du plus au moins fréquent"""
    total = charger_donnees_finales()
    comptage = total['cause_accident_humain'].value_counts()
    return comptage

def cause_accident_humain_global():
    """Compte le nombre d'accidents pour chaque cause d'accident humain, triée du plus au moins fréquent"""
    total = charger_donnees_finales()
    comptage = total['cause_accident_humain'].value_counts()
    return comptage
 
def graphique_cause_accident_humain(total):
    labels_enroules = [textwrap.fill(label, width=35) for label in total.index[::-1]]
 
    plt.figure(figsize=(12, 8))
    bars = plt.barh(labels_enroules, total.values[::-1], color='blue')
    plt.title("Nombre d'accidents par cause d'accident humain", fontsize=14)
    plt.xlabel("Nombre d'accidents", fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.yticks(fontsize=9)
 
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.0, f'{int(width)}', ha='left', va='center', fontweight='bold')
    plt.tight_layout()
    plt.show()
 
def accidents_par_annee_pour_cause(cause:str):
    """Compte, pour une seule cause d'accident humain donnée, le nombre d'accidents par année"""
    total = charger_donnees_finales()
    total = total[total["cause_accident_humain"] == cause]
    comptage = total.groupby("annee").size()
    return comptage
 
def accidents_par_annee_cause():
    """Compte le nombre d'accidents par année, pour chaque cause d'accident humain (une colonne par cause)"""
    total = charger_donnees_finales()
    comptage = total.groupby(["annee", "cause_accident_humain"]).size().unstack(fill_value=0)
    return comptage
 
def graphique_cause_annee(comptage):
    """Affiche, pour chaque cause d'accident humain, l'évolution du nombre d'accidents par année (une courbe par cause)"""
    plt.figure(figsize=(12, 6))
 
    for cause in comptage.columns:
        plt.plot(
            comptage.index,
            comptage[cause],
            marker="o",
            label=cause[:40] + "..." if len(cause) > 40 else cause  # légende raccourcie, sinon illisible
        )
 
    plt.title("Nombre d'accidents par cause humaine et par année")
    plt.xlabel("Année")
    plt.ylabel("Nombre d'accidents")
    plt.grid(True)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))  # légende sortie du graphique (noms longs)
    plt.tight_layout()
    plt.show()

# graphique_type_accident(type_accident_global())
# graphique_type_annee(accidents_par_annee_type())
graphique_cause_accident_humain(cause_accident_humain_global())
graphique_cause_annee(accidents_par_annee_cause())