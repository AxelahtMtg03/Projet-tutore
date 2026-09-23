# Module de prédiction

Ce dossier regroupe les éléments utilisés pour la prédiction à partir de deux approches :

- **Linear Regression** : modèle de régression linéaire servant de référence pour estimer la variable cible à partir des données disponibles ;
- **Random Forest** : modèles et scripts dont le nom commence par `rf`, utilisés pour entraîner et exploiter une forêt aléatoire.

## Organisation

- `main` : point d’entrée du module. Il prépare les données, lance l’entraînement et permet d’obtenir les prédictions avec les modèles retenus ;
- fichiers liés à **Linear Regression** : définition, entraînement et utilisation du modèle de régression linéaire ;
- fichiers commençant par `rf` : code consacré aux modèles Random Forest, à leur entraînement, à leur évaluation ou à la génération des prédictions.

Les autres fichiers et méthodes du dossier ne sont pas concernés par ce module de référence.

## Fonctionnement général

1. Charger et préparer les données ;
2. Séparer les variables explicatives de la variable cible ;
3. Entraîner le modèle de régression linéaire ou le modèle Random Forest ;
4. Produire les prédictions ;
5. Comparer les résultats à l’aide des métriques prévues dans le projet.

## Utilisation

Depuis la racine du projet, exécuter le point d’entrée `main` avec l’environnement Python du projet :

```bash
python src/prediction/main.py
```

Selon l’organisation exacte des fichiers, le nom du fichier d’entrée peut différer légèrement. Dans ce cas, lancer le fichier `main` présent dans ce dossier.

## Remarques

- Vérifier que les données nécessaires sont disponibles avant l’exécution ;
- conserver le même prétraitement entre l’entraînement et la prédiction ;
- les fichiers `rf*` concernent uniquement l’approche Random Forest ;
- la régression linéaire constitue une base simple pour interpréter et comparer les performances des modèles.
