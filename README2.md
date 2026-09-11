# Mémo personnel — projet tutoré

## Le projet en une phrase

Le projet étudie les accidents de navires en Europe pour repérer les zones les plus accidentogènes, comprendre leur évolution dans le temps et produire des prédictions spatiales de risque jusqu'en 2030.

## Question traitée

> Dans quelle mesure l'analyse spatiale et temporelle des accidents maritimes permet-elle d'identifier des zones à risque et d'anticiper leur évolution ?

L'idée centrale est de relier les accidents historiques à la densité de trafic maritime, puis de représenter le résultat sur des cartes interactives.

## Données utilisées

| Jeu de données | Période | Rôle |
| --- | --- | --- |
| Accidents maritimes | 2011–2025 | Localisation, date, heure, gravité, type de navire, port et type d'accident. |
| Densité de trafic AIS | 2017–2021 | Mesure de l'intensité du trafic par point / zone maritime. |
| Flotte mondiale | 2011–2026 | Nombre de navires par zone géographique, année et type de navire. |

Les fichiers bruts sont dans `data/raw/` : ils ne doivent pas être modifiés. Les fichiers prêts à l'analyse et les sorties des modèles sont dans `data/processed/`.

Le fichier principal des accidents nettoyés est `data/processed/maritime_accidents.csv`. Il contient notamment `lat`, `long`, `annee`, `heure`, `saison`, `gravite`, `bateau`, `type_accident` et `cause_accident_humain`.

## Pipeline du projet

```text
Données brutes
    ↓
Nettoyage et mise en forme (`src/data_processing/`)
    ↓
Données traitées (`data/processed/`)
    ↓
Analyses / graphiques et cartes (`src/visualization/`, `src/maps/`)
    ↓
Modèles de prédiction (`src/prediction/`)
    ↓
CSV de prédictions + cartes HTML dans `maps/`
```

## Ce que fait chaque dossier

- `data/raw/` : sources d'origine : accidents, densité AIS et flotte mondiale.
- `data/processed/` : données nettoyées, tendances calculées, prédictions et fichiers de comparaison avec le réel.
- `src/data_processing/` : nettoyage des accidents et de la densité, transformation de la flotte et fonctions de chargement.
- `src/visualization/accidents/` : graphiques par année, heure, saison, gravité, port, type de navire ou d'accident.
- `src/maps/` : génération des cartes Folium (accidents, densité, flotte et prédictions).
- `src/prediction/` : modèles et validation des prédictions.
- `visualization/` : images PNG générées par les analyses.
- `maps/` : cartes HTML à ouvrir dans un navigateur.
- `rapport/demo.tex` et `rapport/demo.pdf` : rapport du projet.

## Analyses déjà présentes

Les scripts permettent d'étudier :

- l'évolution annuelle des accidents ;
- leur répartition selon l'heure et la saison ;
- la gravité, le type d'accident et le type de navire ;
- les ports les plus concernés ;
- des croisements entre plusieurs variables ;
- l'évolution de la flotte mondiale ;
- la densité du trafic maritime.

Les résultats graphiques se trouvent surtout dans `visualization/accidents/`, `visualization/fleet/` et `visualization/prediction/`.

## Cartes disponibles

Les cartes sont des fichiers `.html` ouvrables directement dans un navigateur.

| Dossier | Contenu |
| --- | --- |
| `maps/maritime_accidents_maps/` | Carte des accidents, heatmaps, animations temporelles, grille, zones à risque et filtres. |
| `maps/vessel_density_maps/` | Représentations de la densité de trafic. |
| `maps/fleet_maps/` | Carte de la flotte mondiale. |
| `maps/prediction_map/` | Prédictions, tendances et carte des erreurs. |

## Prédiction : à retenir

Deux approches Random Forest sont utilisées :

1. **Régression** — estime le nombre d'accidents par zone et par année. Le script est `src/prediction/random_forest.py` et produit `predictions_random_forest_densite_2023_2030.csv`.
2. **Classification de tendance** — classe chaque zone selon cinq évolutions : forte diminution, faible diminution, stable, faible augmentation ou forte augmentation. Le script est `src/prediction/random_forest2.py`.

Le modèle de régression exploite des statistiques de densité (moyenne, médiane, maximum, écart-type et nombre de mesures) ainsi qu'une référence historique des accidents. Les résultats sont agrégés sur une grille de 1,5° × 1,5° avant d'être cartographiés.

Les fichiers de comparaison prédiction / réalité sont générés par `src/prediction/test.py`.

## Relancer le projet

Il n'y a pas encore de point d'entrée unique ni de fichier `requirements.txt`. Les bibliothèques utilisées sont notamment : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `folium`, `branca`, `pyproj`, `scipy` et, pour une carte, `requests`.

Depuis la racine du projet, lancer un script se fait par exemple avec :

```powershell
python src/data_processing/accident_processing.py
python src/data_processing/density_processing.py
python src/prediction/linear_regression_accident.py
```

Plusieurs scripts sont organisés sous forme de fonctions et ne s'exécutent pas automatiquement. Dans ce cas, il faut appeler la fonction voulue (par exemple `run_regression_prediction()` dans `src/prediction/random_forest.py`, ou les fonctions de génération dans `src/maps/`).

## Résultats et observations à présenter

- Les accidents se concentrent dans des zones maritimes fortement fréquentées, notamment autour de la Manche, de la Méditerranée et de l'Atlantique nord.
- Les graphiques du projet mettent en évidence une hausse jusqu'au milieu des années 2010, un plateau, puis une diminution progressive sur les années récentes.
- Les cartes permettent de voir à la fois les zones avec le plus d'accidents, les tendances prévues et les zones où l'erreur de prédiction est élevée.

## Limites importantes

- La couverture temporelle des trois sources n'est pas la même : les accidents vont de 2011 à 2025, mais les données AIS s'arrêtent en 2021.
- Les projections futures de densité sont donc une extrapolation et doivent être interprétées avec prudence.
- La grille de 1,5° est adaptée à une vision régionale, pas à un diagnostic précis au niveau d'un port ou d'un chenal.
- Des variables potentiellement importantes manquent : météo, courants, réglementation, trafic portuaire détaillé et évolution technologique.
- Les données d'accidents ont des périodes incomplètes : une partie de 2011 est absente et décembre 2025 n'est pas couvert.

## Pistes pour la suite

1. Créer un `requirements.txt` et un script principal pour relancer le projet facilement.
2. Documenter les fonctions qui doivent être appelées manuellement.
3. Ajouter des données AIS plus récentes et, si possible, des données météo / courants.
4. Tester une grille plus fine et comparer les performances.
5. Créer un tableau de bord Streamlit pour présenter les cartes et graphiques au même endroit.

## Phrase simple pour présenter le projet à l'oral

« Nous avons nettoyé et croisé des données d'accidents maritimes, de trafic AIS et de flotte mondiale afin de cartographier les zones accidentogènes en Europe et d'estimer leur évolution grâce à des modèles Random Forest. »

---

Ce fichier est un mémo de travail personnel. Le README public du projet reste `readme.md`.
