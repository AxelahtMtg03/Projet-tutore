# Prédiction des accidents maritimes

Ce dossier contient plusieurs approches de prédiction des accidents maritimes. Les modèles travaillent par zone géographique : une zone est une cellule de grille de **1,5° × 1,5°** (`TAILLE_ZONE_DEG`). Les accidents trop éloignés d'une cellule de densité de navires (plus de **50 km**) sont écartés.

Les scripts lisent principalement les fichiers suivants dans `data/processed/` :

- `maritime_accidents.csv` : accidents, dates et coordonnées ;
- `vessel_density.csv` : densité de navires (`vd`) par lieu et date ;
- `global_fleet.csv` : taille annuelle de la flotte mondiale.

## Les scripts

| Fichier | Rôle | Résultat principal |
|---|---|---|
| `linear_regression_accident.py` | Régression linéaire très simple sur le nombre total d'accidents par année. Elle utilise seulement l'année et valide le modèle sur 2023–2025. | Graphique `visualization/prediction/validation_regression.png` et métriques MAE/RMSE. |
| `random_forest.py` | Random Forest de **régression** : prédit le nombre d'accidents dans chaque cellule de densité, puis agrège les cellules dans une grille géographique. | `predictions_random_forest_densite_2023_2030.csv`. |
| `random_forest_years.py` | Random Forest de **classification annuelle** : prédit une catégorie de tendance pour chaque zone et année. Les hyperparamètres sont sélectionnés par validation temporelle. | `predictions_tendance_cv_2023_2030.csv`. |
| `random_forest_mensuel.py` | Classification de la tendance par mois et par zone, de 2023 à 2030. | `predictions_tendance_mensuelle_2023_2030.csv`. |
| `random_forest_trimestre.py` | Classification de la tendance par trimestre (T1 à T4) et par zone. | `predictions_tendance_trimestrielle_2023_2030.csv`. |
| `random_forest_semestre.py` | Classification de la tendance par semestre (S1/S2) et par zone, avec sélection temporelle des hyperparamètres. | `predictions_tendance_semestrielle_2023_2030.csv`. |
| `test.py` | Ne crée pas de modèle : compare les fichiers de prédictions aux accidents réellement observés (annuel, mensuel, trimestriel, semestriel et CV). | Fichiers `comparaison_prediction_reel*.csv` et taux d'exactitude affichés. |

## Ce que prédisent les modèles de classification

La cible est `categorie`/`tendance`, déterminée en comparant le prochain créneau au niveau historique de la zone :

- `Forte diminution` : baisse inférieure à -30 % ;
- `Faible diminution` : de -30 % à -10 % ;
- `Stable` : variation faible (ou écart absolu inférieur à 1,5 accident) ;
- `Faible augmentation` : de +10 % à +30 % ;
- `Forte augmentation` : au-dessus de +30 % ;
- `Données insuffisantes` : référence historique trop faible (moins de 0,5 accident).

`confiance` dans les CSV correspond à la probabilité de la catégorie retenue par le classifieur, en pourcentage. Ce n'est pas une garantie que la prédiction est correcte.

## Variables utilisées par les modèles

Les noms `lag` correspondent toujours à l'historique de **la même zone**. Par exemple, `nb_accidents_lag_1` est le nombre d'accidents au créneau précédent.

| Variable | Signification |
|---|---|
| `annee` | Année du créneau à prédire. |
| `mois_sin`, `mois_cos` | Encodage cyclique du mois : janvier et décembre restent proches, contrairement aux nombres 1 et 12. |
| `trimestre` / `semestre` | Position du créneau dans l'année : 1–4 ou 1–2. |
| `densite_moyenne` | Moyenne de `vd`, la densité de navires dans la zone pendant le créneau. |
| `densite_mediane` | Valeur centrale de la densité ; elle est moins influencée par les pics extrêmes. |
| `densite_max` | Plus forte densité de navires observée dans la zone. |
| `densite_std` | Variabilité de la densité de navires pendant le créneau. |
| `nb_mesures` | Nombre de relevés de densité disponibles. |
| `nombre_navires` | Nombre de navires de la ligne `Flotte totale` pour l'année. |
| `nb_accidents_lag_N` | Nombre d'accidents il y a `N` créneaux. `lag_12` mensuel signifie le même mois de l'année précédente ; `lag_12` trimestriel remonte à trois ans. |
| `moyenne_N` | Moyenne des `N` créneaux précédents. |
| `mediane_N` | Médiane des `N` créneaux précédents, plus robuste aux valeurs exceptionnelles qu'une moyenne. |
| `evolution_recente` | Écart en pourcentage entre une statistique courte et une statistique longue ; indique l'accélération ou le ralentissement récent. |
| `reference_historique` | Nombre annuel moyen d'accidents historiquement observé dans une cellule ; utilisé seulement par `random_forest.py`. |

### Variables par modèle

- **Annuel** (`random_forest_years.py`) : `annee`, densité moyenne/maximale, taille de flotte, `lag_1` à `lag_5`, `mediane_3`, `mediane_5`, `evolution_recente`.
- **Mensuel** : `annee`, `mois_sin`, `mois_cos`, densité moyenne/maximale, taille de flotte, `lag_1`, `lag_2`, `lag_3`, `lag_12`, `moyenne_3`, `moyenne_12`, `evolution_recente`.
- **Trimestriel** : `annee`, `trimestre`, densité moyenne/maximale, taille de flotte, `lag_1` à `lag_4`, `lag_12`, `moyenne_4`, `moyenne_12`, `evolution_recente`.
- **Semestriel** : `annee`, `semestre`, densité moyenne/maximale, taille de flotte, `lag_1` à `lag_6`, `mediane_2`, `mediane_4`, `mediane_6`, `evolution_recente`.
- **Régression par densité** (`random_forest.py`) : `densite_moyenne`, `densite_mediane`, `densite_max`, `densite_std`, `nb_mesures`, `reference_historique`. La cible est le nombre exact `nb_accidents`, et non une catégorie.

Les variables comme `reference_*`, `accidents_suivants`, `reference_suivante`, `ecart_absolu_suivant` et `evolution_suivante` servent à construire la cible historique `categorie`. Elles ne sont pas fournies au classifieur comme features.

## Paramètres Random Forest

Une Random Forest construit de nombreux arbres de décision sur des sous-échantillons des données et agrège leur résultat : vote majoritaire pour la classification, moyenne pour la régression.

| Paramètre | Valeur dans le projet | Effet |
|---|---|---|
| `n_estimators` | 200 pour les modèles mensuel/trimestriel ; 300 pour la régression ; 100, 200 ou 300 testés par les modèles annuel/semestriel | Nombre d'arbres. Davantage d'arbres stabilise généralement la prédiction, mais augmente le temps de calcul. |
| `max_depth` | 10 pour les modèles à valeur fixe ; 8, 10 ou 15 testés par les modèles annuel/semestriel | Profondeur maximale d'un arbre. Une profondeur élevée capte plus de détails mais peut surapprendre. |
| `min_samples_split` | 5 pour mensuel/trimestriel ; 2 ou 5 testés par annuel/semestriel | Nombre minimal d'exemples requis pour séparer un nœud. Une valeur plus grande rend l'arbre moins complexe. |
| `min_samples_leaf` | 2 pour mensuel/trimestriel/régression ; 1 ou 2 testés par annuel/semestriel | Nombre minimal d'exemples dans une feuille. Évite des règles basées sur un cas isolé. |
| `class_weight="balanced_subsample"` | Tous les classifieurs | Rééquilibre les classes dans chaque arbre : les catégories rares ont davantage de poids. Non utilisé en régression. |
| `random_state=42` | Tous les modèles | Graine aléatoire fixe : deux exécutions avec les mêmes données donnent le même résultat. |
| `n_jobs=-1` | Tous les modèles | Utilise tous les cœurs processeur disponibles pour accélérer l'entraînement. |

### Sélection automatique des paramètres

Les scripts annuel et semestriel utilisent `GridSearchCV` avec `TimeSeriesSplit(n_splits=5)` : ils essaient les 36 combinaisons du tableau de paramètres et conservent celle ayant la meilleure `accuracy`. La séparation est chronologique : le modèle est entraîné sur le passé et évalué sur des années ultérieures, ce qui évite de lui montrer le futur pendant l'entraînement.

Les scripts mensuel et trimestriel ont des paramètres fixés directement dans le code ; ils n'exécutent pas de recherche par grille.

## Points d'attention

- Les prédictions futures réutilisent la dernière densité et la dernière taille de flotte connues : ce ne sont pas des prévisions indépendantes de trafic maritime.
- Après la dernière période réellement connue, les modèles réinjectent leurs propres prédictions dans l'historique pour produire les périodes suivantes. L'incertitude peut donc s'accumuler jusqu'en 2030.
- Les données manquantes de densité sont complétées avec la dernière/première valeur de la zone, puis si nécessaire avec la moyenne globale.
