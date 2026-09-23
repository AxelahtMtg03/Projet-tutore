"""
Briques communes aux modeles Random Forest de prediction de tendance
(annuel, semestre, trimestre, mois).

Contenu :
  - les 3 classes (Diminution / Stable / Augmentation) et leur categorisation
  - EnsembleSousClasses : la classe majoritaire (Stable) est divisee en n
    sous-classes C1..Cn, un modele est entraine par sous-classe (Ci + toutes
    les autres classes) et les probabilites sont moyennees
  - cross_validation_temporelle : CV "expanding window" par periode
  - resumer_comparaison / sauvegarder_csv : affichage + sauvegarde des resultats
"""
import itertools

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

# ============================================================
# CLASSES
# ============================================================
AUGMENTATION = "Augmentation"
DIMINUTION = "Diminution"
STABLE = "Stable"
INSUFFISANT = "Données insuffisantes"
CLASSES = [DIMINUTION, STABLE, AUGMENTATION]

TAILLE_ZONE_DEG = 1.5
DISTANCE_MAX_M = 50_000    # rayon max entre un accident et la cellule de densite la plus proche
REFERENCE_MIN = 0.5       # en dessous : "Données insuffisantes"
SEUIL_ECART_ABS = 1.5     # |ecart| < 1.5 accident  -> Stable
SEUIL_PCT = 10            # +/- 10 %                -> Stable

# Variation representative utilisee pour projeter le nombre d'accidents
# au-dela des donnees reelles (avant : faible = +/-15, forte = +/-40)
DELTA_REPRESENTATIF = {
    DIMINUTION: -25,
    STABLE: 0,
    AUGMENTATION: 25,
    INSUFFISANT: 0,
}

N_MAX_SOUS_CLASSES = 10

# True : la decision est reglee sur la CV (poids par classe) pour mieux detecter Augmentation/Diminution,
#        au prix d'une accuracy globale plus basse. False : argmax simple des probabilites.
AJUSTER_POIDS = True
def configurer_seuils(ecart=None, pct=None, ref=None):
    global SEUIL_ECART_ABS, SEUIL_PCT, REFERENCE_MIN
    if ecart is not None: SEUIL_ECART_ABS = ecart
    if pct is not None:   SEUIL_PCT = pct
    if ref is not None:   REFERENCE_MIN = ref

def calculer_zone(lat, lon, taille=TAILLE_ZONE_DEG):
    zone_lat = np.floor(lat / taille) * taille
    zone_lon = np.floor(lon / taille) * taille
    return "z_" + zone_lat.round(1).astype(str) + "_" + zone_lon.round(1).astype(str)


def categoriser(ecart_absolu, pct, reference):
    """Categorise en 3 classes : Diminution / Stable / Augmentation."""
    if pd.isna(pct) or pd.isna(reference):
        return np.nan
    if reference < REFERENCE_MIN:
        return INSUFFISANT
    if abs(ecart_absolu) < SEUIL_ECART_ABS:
        return STABLE
    if pct < -SEUIL_PCT:
        return DIMINUTION
    if pct < SEUIL_PCT:
        return STABLE
    return AUGMENTATION


def categoriser_colonnes(ecart_absolu, pct, reference):
    """Version 'colonnes' de categoriser (plus rapide qu'un DataFrame.apply)."""
    return [categoriser(e, p, r) for e, p, r in zip(ecart_absolu, pct, reference)]


def assigner_zones_densite(accidents, density, distance_max_m=DISTANCE_MAX_M):
    """
    Rattache chaque accident a la cellule de densite la plus proche (<= distance_max_m),
    puis nomme la zone (grille de 1.5 deg) a partir de la position de cette cellule.
    Renvoie (accidents avec colonne 'zone', zones_natives).
    """
    accidents = accidents.copy()
    vers_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_acc, y_acc = vers_3035.transform(accidents["long"].values, accidents["lat"].values)
    accidents["x"] = x_acc
    accidents["y"] = y_acc

    zones_natives = density[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    zones_natives["zone_id"] = zones_natives.index
    arbre = cKDTree(zones_natives[["latitude", "longitude"]].values)

    distances, plus_proche = arbre.query(accidents[["y", "x"]].values)
    accidents["zone_id"] = zones_natives.loc[plus_proche, "zone_id"].values
    accidents["distance_zone"] = distances
    accidents = accidents[accidents["distance_zone"] <= distance_max_m].copy()

    vers_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = vers_4326.transform(zones_natives["longitude"].values,
                                           zones_natives["latitude"].values)
    zones_natives["lat_deg"] = lat_deg
    zones_natives["lon_deg"] = lon_deg
    zones_natives["zone"] = calculer_zone(zones_natives["lat_deg"], zones_natives["lon_deg"])

    accidents = accidents.merge(zones_natives[["zone_id", "zone"]], on="zone_id", how="left")
    return accidents, zones_natives


def afficher_distribution(train, colonne="categorie"):
    print("\nDistribution de la cible (3 classes) :")
    comptes = train[colonne].value_counts()
    for classe in CLASSES:
        n = int(comptes.get(classe, 0))
        print(f"  {classe}: {n} ({n / len(train) * 100:.1f}%)")


def calculer_reference_saisonniere(agg, sous_cols, annees_cibles, derniere_annee_observee):
    """
    Reference IDENTIQUE a celle du test : mediane des comptes non nuls de la meme
    (zone, sous-periode) sur les annees connues, jamais apres derniere_annee_observee.

    Pour une annee cible Y, seules les annees < Y sont utilisees : a l'entrainement il n'y a
    donc pas de fuite, et pour Y > derniere_annee_observee on retrouve exactement la reference
    du test (toutes les annees <= derniere_annee_observee).

    agg        : DataFrame [zone, annee, *sous_cols, nb_accidents] (toutes les annees dispo)
    sous_cols  : [] (annuel), ["semestre"], ["trimestre"] ou ["mois_num"]
    Renvoie    : [zone, *sous_cols, annee, ref_hist, nb_annees_hist]
    """
    a = agg[(agg["nb_accidents"] > 0) & (agg["annee"] <= derniere_annee_observee)]
    lignes = []
    for cle, g in a.groupby(["zone"] + list(sous_cols)):
        if not isinstance(cle, tuple):
            cle = (cle,)
        g = g.sort_values("annee")
        annees = g["annee"].values
        valeurs = g["nb_accidents"].values.astype(float)
        for annee_cible in annees_cibles:
            n = int((annees < annee_cible).sum())
            if n:
                lignes.append(cle + (annee_cible, float(np.median(valeurs[:n])), n))
    return pd.DataFrame(lignes, columns=["zone"] + list(sous_cols) + ["annee", "ref_hist", "nb_annees_hist"])


# ============================================================
# ENSEMBLE PAR SOUS-CLASSES DE LA CLASSE MAJORITAIRE
# ============================================================
class EnsembleSousClasses:
    """
    La classe majoritaire (Stable) est repartie au hasard en n sous-classes
    C1..Cn. Pour chaque Ci, un Random Forest est entraine sur
    Ci + toutes les autres classes (Augmentation, Diminution). Les
    probabilites des n modeles sont ensuite moyennees.

    n_sous_classes="auto" : n est choisi pour que chaque Ci ait a peu pres la
    taille de la plus grosse classe minoritaire (borne entre 1 et N_MAX).
    """

    def __init__(self, n_sous_classes="auto", n_estimators=200, max_depth=10,
                 min_samples_split=5, min_samples_leaf=2, random_state=42,
                 n_jobs=-1, verbose=False):
        self.n_sous_classes = n_sous_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _choisir_n(self, comptes, classe_maj):
        n_maj = comptes[classe_maj]
        if self.n_sous_classes != "auto":
            n = int(self.n_sous_classes)
        else:
            minoritaires = comptes.drop(classe_maj)
            if len(minoritaires) == 0:
                return 1
            n = int(round(n_maj / minoritaires.max()))
            n = min(n, N_MAX_SOUS_CLASSES)
        return max(1, min(n, n_maj))

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)

        self.classes_ = np.array(sorted(y.unique()))
        self.poids_ = np.ones(len(self.classes_))   # regle de decision (voir optimiser_poids_classes)
        comptes = y.value_counts()
        self.classe_majoritaire_ = comptes.idxmax()

        idx_maj = np.where(y == self.classe_majoritaire_)[0]
        idx_autres = np.where(y != self.classe_majoritaire_)[0]

        n = self._choisir_n(comptes, self.classe_majoritaire_)
        rng = np.random.RandomState(self.random_state)
        sous_classes = np.array_split(rng.permutation(idx_maj), n)

        if self.verbose:
            tailles = [len(c) for c in sous_classes]
            print(f"  Classe majoritaire '{self.classe_majoritaire_}' : {len(idx_maj)} lignes "
                  f"-> {n} sous-classes (C1..C{n}) de {min(tailles)} a {max(tailles)} lignes")
            print(f"  Autres classes gardees entieres dans chaque modele : "
                  f"{ {c: int(v) for c, v in comptes.drop(self.classe_majoritaire_).items()} }")

        self.modeles_ = []
        for i, idx_ci in enumerate(sous_classes):
            idx = np.concatenate([idx_ci, idx_autres])
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced", random_state=self.random_state + i,
                n_jobs=self.n_jobs,
            )
            rf.fit(X.iloc[idx], y.iloc[idx])
            self.modeles_.append(rf)
        return self

    def predict_proba(self, X):
        """Moyenne des probabilites des n modeles (colonnes = self.classes_)."""
        proba = np.zeros((len(X), len(self.classes_)))
        for rf in self.modeles_:
            p = rf.predict_proba(X)
            for j, classe in enumerate(rf.classes_):
                proba[:, np.searchsorted(self.classes_, classe)] += p[:, j]
        return proba / len(self.modeles_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X) * self.poids_, axis=1)]


# ============================================================
# CROSS-VALIDATION TEMPORELLE (expanding window)
# ============================================================
def cross_validation_temporelle(data, features, colonne_cible, colonne_temps="annee",
                                colonne_groupe=None, avec_oof=False, **params_modele):
    """
    Pour chaque groupe G (a partir du 2e) : entrainement sur tout ce qui est
    strictement avant G, validation sur G. Le futur n'est jamais melange au passe,
    contrairement a un KFold classique.

    colonne_temps  : colonne triable qui ordonne les lignes (annee, index de periode, mois...)
    colonne_groupe : bloc de validation (par defaut = colonne_temps). Pour les modeles
                     infra-annuels on prend l'annee : 1 fold = 1 annee de validation.
    avec_oof       : si True, renvoie aussi les probabilites hors-echantillon de tous les
                     folds (dict y / proba / classes) pour regler la decision.
    """
    colonne_groupe = colonne_groupe or colonne_temps
    groupes = sorted(data[colonne_groupe].unique())
    classes_globales = np.array(sorted(data[colonne_cible].unique()))
    lignes, y_oof, p_oof = [], [], []
    for k in range(1, len(groupes)):
        g_val = groupes[k]
        val = data[data[colonne_groupe] == g_val]
        train = data[data[colonne_temps] < val[colonne_temps].min()]
        if len(val) == 0 or len(train) == 0 or train[colonne_cible].nunique() < 2:
            continue

        modele = EnsembleSousClasses(**params_modele).fit(train[features], train[colonne_cible])
        proba = modele.predict_proba(val[features])
        y_pred = modele.classes_[np.argmax(proba, axis=1)]
        y_vrai = val[colonne_cible].values

        # probas ramenees sur l'ensemble des classes (une classe peut manquer dans un fold)
        p_glob = np.zeros((len(val), len(classes_globales)))
        for j, classe in enumerate(modele.classes_):
            p_glob[:, np.searchsorted(classes_globales, classe)] = proba[:, j]
        y_oof.append(y_vrai)
        p_oof.append(p_glob)

        lignes.append({
            "fold": len(lignes) + 1,
            "periodes_train": f"{train[colonne_groupe].min()}-{train[colonne_groupe].max()}",
            "periode_val": g_val,
            "n_train": len(train),
            "n_val": len(val),
            "accuracy": round((y_pred == y_vrai).mean() * 100, 1),
            "balanced_accuracy": round(balanced_accuracy_score(y_vrai, y_pred) * 100, 1),
            "f1_macro": round(f1_score(y_vrai, y_pred, average="macro", zero_division=0) * 100, 1),
        })

    cv_df = pd.DataFrame(lignes)
    if not avec_oof:
        return cv_df
    oof = {
        "y": np.concatenate(y_oof) if y_oof else np.array([]),
        "proba": np.vstack(p_oof) if p_oof else np.zeros((0, len(classes_globales))),
        "classes": classes_globales,
    }
    return cv_df, oof


def optimiser_poids_classes(oof, classe_ref=STABLE,
                            grille=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0), gain_min=0.5):
    """
    Regle de decision : on multiplie les probabilites de chaque classe par un poids avant
    de prendre l'argmax. Les poids (Stable = 1 fixe) sont choisis sur les probabilites
    hors-echantillon de la CV pour maximiser la balanced accuracy (moyenne des rappels).
    Corrige la tendance du modele a repondre "Stable" partout.
    Si le gain est inferieur a gain_min point, on garde des poids neutres (= 1).
    """
    classes, y, proba = oof["classes"], oof["y"], oof["proba"]
    neutre = np.ones(len(classes))
    if len(y) == 0 or len(classes) < 2:
        return neutre
    if not AJUSTER_POIDS:
        return neutre

    def score(w):
        return balanced_accuracy_score(y, classes[np.argmax(proba * w, axis=1)]) * 100

    libres = [i for i, c in enumerate(classes) if c != classe_ref]
    base = score(neutre)
    meilleur, meilleur_score = neutre, base
    for combo in itertools.product(grille, repeat=len(libres)):
        w = np.ones(len(classes))
        for i, v in zip(libres, combo):
            w[i] = v
        sc = score(w)
        # a score egal, on prefere les poids les plus proches de 1
        if sc > meilleur_score + 1e-9 or (abs(sc - meilleur_score) < 1e-9
                                          and np.abs(np.log(w)).sum() < np.abs(np.log(meilleur)).sum()):
            meilleur, meilleur_score = w, sc

    print("\n" + "-" * 60)
    print("REGLAGE DE LA DECISION (poids par classe, choisis sur la CV)")
    print("-" * 60)
    if meilleur_score - base < gain_min:
        print(f"  Gain de balanced accuracy trop faible ({meilleur_score - base:+.1f} pt) : poids neutres.")
        return neutre
    print("  Poids : " + ", ".join(f"{c}={w:g}" for c, w in zip(classes, meilleur)))
    print(f"  Balanced accuracy en CV : {base:.1f}% -> {meilleur_score:.1f}%")
    return meilleur


def afficher_cv(cv_df):
    print("\n" + "-" * 60)
    print("CROSS-VALIDATION TEMPORELLE (train = passe, validation = annee suivante)")
    print("-" * 60)
    if cv_df.empty:
        print("  Pas assez de periodes pour faire une cross-validation.")
        return
    print(cv_df.to_string(index=False))
    print(f"\n  Moyenne sur {len(cv_df)} folds : "
          f"accuracy = {cv_df['accuracy'].mean():.1f}% | "
          f"balanced accuracy = {cv_df['balanced_accuracy'].mean():.1f}% | "
          f"F1 macro = {cv_df['f1_macro'].mean():.1f}%")


def metriques_cv(cv_df, granularite):
    """CV -> lignes au format long (granularite, section, detail, metrique, valeur, n)."""
    lignes = []
    for _, r in cv_df.iterrows():
        for m in ["accuracy", "balanced_accuracy", "f1_macro"]:
            lignes.append({"granularite": granularite, "section": "cv_temporelle",
                           "detail": f"fold_{int(r['fold'])}_val_{r['periode_val']}",
                           "metrique": m, "valeur": r[m], "n": int(r["n_val"])})
    for m in ["accuracy", "balanced_accuracy", "f1_macro"]:
        if not cv_df.empty:
            lignes.append({"granularite": granularite, "section": "cv_temporelle",
                           "detail": "moyenne", "metrique": m,
                           "valeur": round(cv_df[m].mean(), 1), "n": len(cv_df)})
    return pd.DataFrame(lignes)


# ============================================================
# RESUME + SAUVEGARDE
# ============================================================
def resumer_comparaison(df_compare, granularite, titre, colonne_periode="annee"):
    """
    Affiche le resume de la comparaison predit / reel (meme format qu'avant)
    et renvoie les metriques au format long pour le CSV.
    """
    lignes = []

    def ajouter(section, detail, metrique, valeur, n):
        lignes.append({"granularite": granularite, "section": section, "detail": detail,
                       "metrique": metrique, "valeur": round(valeur, 1), "n": n})

    n_total = len(df_compare)
    n_zones_uniques = df_compare["zone"].nunique()
    acc = df_compare["correct"].mean() * 100
    bal = balanced_accuracy_score(df_compare["tendance_reelle"], df_compare["tendance"]) * 100
    f1 = f1_score(df_compare["tendance_reelle"], df_compare["tendance"],
                  average="macro", zero_division=0) * 100

    print("\n" + "=" * 60)
    print(titre)
    print("=" * 60)
    print(f"\nAccuracy globale: {acc:.1f}%")
    print(f"Accuracy equilibree (balanced): {bal:.1f}%")
    print(f"F1 macro: {f1:.1f}%")
    print(f"Zones comparees: {n_total} lignes ({n_zones_uniques} zones uniques)")

    # Reperes : sans eux, un score eleve peut simplement venir de la classe majoritaire
    repartition = df_compare["tendance_reelle"].value_counts(normalize=True) * 100
    classe_maj, base_maj = repartition.index[0], repartition.iloc[0]
    base_bal = 100 / df_compare["tendance_reelle"].nunique()
    print("Repartition reelle: " + ", ".join(f"{c} {repartition.get(c, 0):.1f}%" for c in CLASSES))
    print(f"Repere 'toujours {classe_maj}': accuracy {base_maj:.1f}% | balanced {base_bal:.1f}%")
    print(f"Gain du modele vs repere: accuracy {acc - base_maj:+.1f} pts | balanced {bal - base_bal:+.1f} pts")
    ajouter("test", "globale", "accuracy", acc, n_total)
    ajouter("test", "globale", "balanced_accuracy", bal, n_total)
    ajouter("test", "globale", "f1_macro", f1, n_total)
    ajouter("test", "globale", "n_zones_uniques", n_zones_uniques, n_total)
    ajouter("test", "globale", "repere_toujours_majoritaire_accuracy", base_maj, n_total)
    ajouter("test", "globale", "repere_hasard_balanced", base_bal, n_total)

    print(f"\nRESUME PAR {colonne_periode.upper()}:")
    for p in sorted(df_compare[colonne_periode].unique()):
        d = df_compare[df_compare[colonne_periode] == p]
        a = d["correct"].mean() * 100
        print(f"  {p}: {len(d)} zones, Accuracy = {a:.1f}%")
        ajouter(f"par_{colonne_periode}", str(p), "accuracy", a, len(d))

    print("\nRESUME PAR TENDANCE:")
    for classe in CLASSES:
        d = df_compare[df_compare["tendance_reelle"] == classe]
        if len(d) > 0:
            a = d["correct"].mean() * 100
            print(f"  {classe}: {len(d)} zones, Accuracy = {a:.1f}%")
            ajouter("par_tendance", classe, "accuracy", a, len(d))

    return pd.DataFrame(lignes)


def sauvegarder_csv(df, chemin):
    # utf-8-sig : les accents s'affichent correctement a l'ouverture dans Excel
    df.to_csv(chemin, index=False, encoding="utf-8-sig")
    print(f"CSV sauvegarde: {chemin} ({len(df)} lignes)")