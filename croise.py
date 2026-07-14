from charge import *

def comptage_croise(para1:str,para2:str):
    """Croise deux colonnes (ex: saison/bateau) et compte le nombre d'accidents pour chaque combinaison"""
    total = charger_donnees_finales()
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
    plt.xlabel(para1)
    plt.ylabel("Nombre d'accidents")

    plt.grid(True)
    plt.legend()

    plt.show()
    
def comptage_croise_bis(para1:str, para2:str, para3:str):
    """Comme comptage_croise, mais filtré ex le nb d'accidnet de bateaux de croisière en fonction des saisons"""
    total = charger_donnees_finales()

    total = total[total[para1] == para3]
    comptage = total.groupby(para2).size()
    return comptage
 
def graphique_bis(comptage,para1:str, para2:str, para3:str):
    """Affiche l'évolution du nombre d'accidents selon para2, pour la seule valeur para3 filtrée en amont"""
    plt.figure(figsize=(12,6))
 
    plt.plot(comptage.index, comptage.values, marker="o")
 
    plt.title(f"Nombre d'accidents par {para2}, pour {para1} = \"{para3}\"", fontsize=14)
    plt.xlabel(f"{para2}", fontsize=12)
    plt.ylabel("Nombre d'accidents", fontsize=12)
    plt.grid(True)
 
    plt.show()
 
# b="saison"
# a="bateau"
# graphique((comptage_croise(b,a)),b,a)
 
# Exemple d'utilisation de la version filtrée (les noms de port incluent le pays, ex: "NETHERLANDS - Rotterdam") :
a="bateau"
b="saison"
c= "Cargo ship"
graphique_bis(comptage_croise_bis(a, b,c),a, b, c)