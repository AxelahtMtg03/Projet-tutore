# Données 

Ce dossier contient toutes les données nécésaire pour le projet. Divisé en 2 parties une pour les données brut, donc non modifié, et les données traitées donc après un nettoyage ou bien tout ce qui va etre prédiction.

## Structure

```
data/
├── raw/                          # Données brutes (NE PAS MODIFIER)
│   ├── accidents_mer/            # Données d'accidents maritimes
│   ├── flotte_globale/           # Données de flotte mondiale
│   └── vessel_density/           # Données de densité
│
└── processed/                    # Données traitées
    ├── maritime_accidents.csv
    ├── global_fleet.csv
    └── vessel_density.csv
```

## 📄 Fichiers bruts (`raw/`)

### `accidents_mer/`
- **Description :** Données d'accidents
- **Période :** 01/06/2011 jusqu'au 30/11/2025
- **Colonnes principales :** 
  - `Date of occurrence` : Date de l'accident
  - `Sea area of occurrence` : Zone maritime
  - `State Reporting` : État rapporteur
  - `Competent authority` : Autorité compétente
  - `Casualty Report Nr.` : Numéro de rapport
  - `Occurrence severity` : Gravité
  - `Latitude` : Latitude
  - `Longitude` : Longitude
  - `Third party/other damage` : Dommages tiers
  - `Time (LT) of occurrence` : Heure
  - `Lives lost Occurrence-Total` : Vies perdues
  - `People Injured Occurrence-Total` : Blessés
  - `Port of accident` : Port
  - `Investigation Status` : Statut enquête
  - `Directive 2009/18` : Directive
  - `National location` : Localisation nationale
  - `Missing people Occurrence - Total` : Disparus
  - `Loss / damage to ship or equipment` : Pertes/Dommages
  - `Did the ship sink?` : Naufrage
  - `Pollution (bunkers)` : Pollution (soutes)
  - `Pollution (cargo)` : Pollution (cargaison)
  - `Place on board` : Lieu à bord
  - `Ship operation` : Opération du navire
  - `Poll. quantity/bunker` : Quantité pollution (soutes)
  - `Poll. quantity/cargo` : Quantité pollution (cargaison)
  - `Ship’s routeing` : Routage
  - `Voyage segment` : Segment du voyage
  - `Cargo damage` : Dommages à la cargaison
  - `IMO number` : Numéro IMO
  - `Name of ship` : Nom du navire
  - `Ship / craft type` : Type de navire
  - `Port of departure` : Port de départ
  - `Port of destination` : Port de destination
  - `Occurrence with ship(s)` : Type d'accident
  - `Deviation (CE)` : Cause humaine
  - `Occurrence with person(s)` : Personnes impliquées
  - `Release of pollutants in the air` : Rejets polluants

### `flotte_globale/`
- **Description :**  Données de flotte mondiale
- **Période :** 2011 à 2026
- **Colonnes principales :** 
  - `Economy_Label` : Zone géographique
  - `{année}_{type_navire}_Nombre_de_navires_Value` : Nombre de navires
  - `{année}_{type_navire}_Number_of_ships_Footnote` : Notes
  - `{année}_{type_navire}_Number_of_ships_MissingValue` : Données manquantes
- **Types de navires :** Flotte totale, Pétroliers, Vraquiers, Navires de charge classique, Porte-conteneurs, Autres navires
- **Traitement :** Transformation en format long → `processed/global_fleet.csv`

### `vessel_density/`
- **Description :** Données de densité
- **Période :** 2017 à 2021
- **Colonnes principales :** 
  - `time` : Date et heure
  - `y` : Coordonnée Y 
  - `x` : Coordonnée X 
  - `vd` : Valeur de densité
---

## 📄 Fichiers traités (`processed/`)

### `maritime_accidents.csv`
- **Description :** Données d'accidents nettoyées et enrichies
- **Lignes :** 44 098
- **Colonnes :** 
  - `gravite` : Gravité (catégorie)
  - `long` : Longitude (décimal)
  - `lat` : Latitude (décimal)
  - `heure` : Heure (sans minutes)
  - `port` : Port (nettoyé)
  - `saison` : Saison (Printemps/Été/Automne/Hiver)
  - `annee` : Année
  - `bateau` : Type de bateau (catégorie)
  - `type_accident` : Type d'accident (catégorie)
  - `cause_accident_humain` : Cause humaine (catégorie)

> **Note :** Les colonnes ajoutées (`long`, `lat`, `heure`, etc.) ont été créées lors du traitement pour simplifier l'analyse. Les données d'origine restent inchangées dans `raw/`.


### `global_fleet.csv`
- **Description :** Données de flotte mondiale
- **Lignes :** 21 049
- **Colonnes :**  
  - `economie` : Zone géographique
  - `annee` : Année
  - `type_navire` : Type de navire
  - `nombre_navires` : Nombre de navires

### `vessel_density.csv`
- **Description :** Données de densité
- **Lignes :** 446 761
- **Colonnes :** 
  - `time` : Date et heure
  - `vd` : Valeur de densité
  - `latitude` : Latitude
  - `longitude` : Longitude

### `predictions_random_forest_densite_2023_2030.csv`
- **Description :** Prédictions du modèle Random Forest (régression)
- **Généré par :** `src/prediction/random_forest.py`
- **Colonnes :** 11 961

### `predictions_tendance_2023_2030.csv`
- **Description :** Prédictions du modèle Random Forest (tendance)
- **Généré par :** `src/prediction/random_forest.py`
- **Colonnes :** 20 673

### `comparaison_prediction_reel.csv`
- **Description :** Comparaison entre les prédictions et la réalité
- **Généré par :** `src/prediction/test.py`
- **Colonnes :** 1 422