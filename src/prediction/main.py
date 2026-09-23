"""
Lance les modeles Random Forest (annuel, semestre, trimestre, mois) en meme
temps, affiche leurs prints (prefixes par le nom du modele) et rassemble les
metriques dans un seul CSV de synthese.

Deux modes (configurables en haut du fichier) :
    MODE = "normal"       -> ecrit synthese_tendances.csv
    MODE = "grid_search"  -> teste plusieurs seuils, ecrit :
                                grid_search_resultats.csv (format long)
                                grid_search_pivot.csv     (format large trie)

Format de synthese_tendances.csv :
    granularite,classe,accuracy

Format de grid_search_resultats.csv :
    seuil_ecart,seuil_pct,granularite,classe,accuracy

Format de grid_search_pivot.csv :
    seuil_ecart,seuil_pct,granularite,Diminution,Stable,Augmentation,Global,score_equilibre
"""
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURATION (a modifier manuellement)
# ============================================================
MODE = "grid_search"        # "normal" ou "grid_search"
MODELES_A_LANCER = None     # None = tous, ou une liste comme ["trimestre", "mois"]
SEQUENTIEL = False          # True = un par un, False = en parallele

# Grille pour le grid search (utilise seulement si MODE = "grid_search")
SEUILS_ECART = [0.5, 1.0, 1.5, 2.0]
SEUILS_PCT = [5.0, 10.0, 15.0, 20.0]

# ============================================================
# CONSTANTES
# ============================================================
DOSSIER_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = DOSSIER_SCRIPTS.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
DATA_PATH.mkdir(parents=True, exist_ok=True)

SCRIPTS = {
    "annuel": "rf_years.py",
    "semestre": "rf_semestre.py",
    "trimestre": "rf_trimestre.py",
    "mois": "rf_months.py",
}

NOM_FICHIER = {
    "annuel": "annuelle",
    "semestre": "semestrielle",
    "trimestre": "trimestrielle",
    "mois": "mensuelle",
}
NOM_GRANULARITE = {
    "annuel": "annuel",
    "semestre": "semestriel",
    "trimestre": "trimestriel",
    "mois": "mensuel",
}

ORDRE_CLASSES = ["Diminution", "Stable", "Augmentation"]
ORDRE_GRANULARITES = ["annuel", "semestriel", "trimestriel", "mensuel"]

MOTIF_METRIQUES = "metriques_tendance_*.csv"
FICHIER_SYNTHESE = DATA_PATH / "synthese_tendances.csv"
FICHIER_GRID = DATA_PATH / "grid_search_resultats.csv"
FICHIER_PIVOT = DATA_PATH / "grid_search_pivot.csv"

_verrou_print = threading.Lock()


# ============================================================
# LANCEMENT D'UN SCRIPT
# ============================================================
def lancer_script(nom, script, seuil_ecart=None, seuil_pct=None):
    """Lance un script dans son propre processus et affiche sa sortie en direct."""
    debut = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, str(script)]
    if seuil_ecart is not None and seuil_pct is not None:
        cmd += ["--seuil-ecart", str(float(seuil_ecart)), "--seuil-pct", str(float(seuil_pct))] 
        prefixe = f"{nom} e={seuil_ecart} p={seuil_pct}"
    else:
        prefixe = nom

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        cwd=str(script.parent), env=env,
    )
    for ligne in proc.stdout:
        with _verrou_print:
            print(f"[{prefixe}] {ligne.rstrip()}")
    code = proc.wait()
    return {"modele": nom, "statut": "OK" if code == 0 else f"ERREUR (code {code})",
            "duree_s": round(time.time() - debut, 1)}


# ============================================================
# SYNTHESE (mode normal)
# ============================================================
def construire_synthese(resume):
    """CSV long -> (granularite, classe, accuracy)."""
    masque_global = ((resume["section"] == "test")
                     & (resume["detail"] == "globale")
                     & (resume["metrique"] == "accuracy"))
    masque_tendance = ((resume["section"] == "par_tendance")
                       & (resume["metrique"] == "accuracy"))

    extraits = []
    if masque_global.any():
        g = resume.loc[masque_global, ["granularite", "valeur"]].copy()
        g["classe"] = "Global"
        extraits.append(g)
    if masque_tendance.any():
        t = resume.loc[masque_tendance, ["granularite", "detail", "valeur"]].copy()
        t = t.rename(columns={"detail": "classe"})
        extraits.append(t)

    if not extraits:
        return pd.DataFrame(columns=["granularite", "classe", "accuracy"])

    synthese = pd.concat(extraits, ignore_index=True)
    synthese = synthese.rename(columns={"valeur": "accuracy"})
    synthese["accuracy"] = synthese["accuracy"].round(1)

    ordre_gran = {g: i for i, g in enumerate(ORDRE_GRANULARITES)}
    ordre_cls = {c: i for i, c in enumerate(ORDRE_CLASSES + ["Global"])}
    synthese["_g"] = synthese["granularite"].map(ordre_gran).fillna(99)
    synthese["_c"] = synthese["classe"].map(ordre_cls).fillna(99)
    synthese = (synthese.sort_values(["_g", "_c"])
                .drop(columns=["_g", "_c"])
                .reset_index(drop=True))
    return synthese[["granularite", "classe", "accuracy"]]


def mode_normal(choisis, sequentiel):
    """Lance les modeles choisis et ecrit synthese_tendances.csv."""
    a_lancer = {}
    for nom in choisis:
        chemin = DOSSIER_SCRIPTS / SCRIPTS[nom]
        if chemin.exists():
            a_lancer[nom] = chemin
        else:
            print(f"[!] Script introuvable pour '{nom}' : {chemin.name} (ignore)")

    if not a_lancer:
        print("Aucun script a lancer.")
        return

    print("=" * 70)
    mode = "sequentiel" if sequentiel else "parallele"
    print(f"LANCEMENT ({mode}) : {', '.join(a_lancer)}")
    print("=" * 70)

    debut_global = time.time()
    if sequentiel:
        bilans = [lancer_script(nom, chemin) for nom, chemin in a_lancer.items()]
    else:
        with ThreadPoolExecutor(max_workers=len(a_lancer)) as pool:
            futures = [pool.submit(lancer_script, nom, chemin) for nom, chemin in a_lancer.items()]
            bilans = [f.result() for f in futures]

    print("\n" + "=" * 70)
    print("BILAN")
    print("=" * 70)
    print(pd.DataFrame(bilans).to_string(index=False))
    print(f"Duree totale : {time.time() - debut_global:.1f}s")

    recents = [f for f in DATA_PATH.glob(MOTIF_METRIQUES)
               if f.stat().st_mtime >= debut_global and f != FICHIER_SYNTHESE]
    if not recents:
        print("\nAucun fichier de metriques trouve pour ce lancement.")
        return

    resume = pd.concat(
        [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(recents)],
        ignore_index=True,
    )
    synthese = construire_synthese(resume)
    synthese.to_csv(FICHIER_SYNTHESE, index=False, encoding="utf-8-sig")
    print(f"\nCSV de synthese sauvegarde: {FICHIER_SYNTHESE} ({len(synthese)} lignes)")
    print()
    print(synthese.to_string(index=False))


# ============================================================
# GRID SEARCH
# ============================================================
def extraire_metriques(fichier, granularite, seuil_ecart, seuil_pct):
    """Lit un CSV de metriques et renvoie une liste de lignes (classe, accuracy)."""
    if not fichier.exists():
        return []
    metr = pd.read_csv(fichier, encoding="utf-8-sig")
    lignes = []

    # Global
    v = metr[(metr["section"] == "test")
             & (metr["detail"] == "globale")
             & (metr["metrique"] == "accuracy")]
    if len(v):
        lignes.append({
            "seuil_ecart": seuil_ecart, "seuil_pct": seuil_pct,
            "granularite": granularite, "classe": "Global",
            "accuracy": float(v["valeur"].iloc[0]),
        })

    # Par tendance
    for classe in ORDRE_CLASSES:
        v = metr[(metr["section"] == "par_tendance")
                 & (metr["detail"] == classe)
                 & (metr["metrique"] == "accuracy")]
        if len(v):
            lignes.append({
                "seuil_ecart": seuil_ecart, "seuil_pct": seuil_pct,
                "granularite": granularite, "classe": classe,
                "accuracy": float(v["valeur"].iloc[0]),
            })

    return lignes


def mode_grid_search(choisis, seuils_ecart, seuils_pct):
    """Teste plusieurs combinaisons de seuils et ecrit grid_search_resultats.csv + pivot."""
    a_lancer = {nom: DOSSIER_SCRIPTS / SCRIPTS[nom] for nom in choisis
                if (DOSSIER_SCRIPTS / SCRIPTS[nom]).exists()}
    if not a_lancer:
        print("Aucun script a lancer.")
        return

    total = len(seuils_ecart) * len(seuils_pct)
    i = 0
    resultats = []

    for e in seuils_ecart:
        for p in seuils_pct:
            i += 1
            print("\n" + "=" * 70)
            print(f"COMBINAISON {i}/{total} : seuil_ecart={e}, seuil_pct={p}")
            print("=" * 70)

            with ThreadPoolExecutor(max_workers=len(a_lancer)) as pool:
                futures = [pool.submit(lancer_script, nom, chemin, e, p)
                           for nom, chemin in a_lancer.items()]
                for f in futures:
                    f.result()

            # Lire les CSV de metriques pour cette combinaison
            for nom in a_lancer:
                gran = NOM_GRANULARITE[nom]
                nom_fichier = NOM_FICHIER[nom]
                fichier = DATA_PATH / f"metriques_tendance_{nom_fichier}_subdivision_e{float(e)}_p{float(p)}.csv"
                resultats.extend(extraire_metriques(fichier, gran, e, p))

    if not resultats:
        print("Aucun resultat collecte.")
        return

    df = pd.DataFrame(resultats)

    # Trier : par seuils, puis granularite, puis classe
    ordre_gran = {g: i for i, g in enumerate(ORDRE_GRANULARITES)}
    ordre_cls = {c: i for i, c in enumerate(ORDRE_CLASSES + ["Global"])}
    df["_g"] = df["granularite"].map(ordre_gran).fillna(99)
    df["_c"] = df["classe"].map(ordre_cls).fillna(99)
    df = (df.sort_values(["seuil_ecart", "seuil_pct", "_g", "_c"])
            .drop(columns=["_g", "_c"])
            .reset_index(drop=True))
    df["accuracy"] = df["accuracy"].round(1)

    df.to_csv(FICHIER_GRID, index=False, encoding="utf-8-sig")
    print(f"\nCSV grid search sauvegarde: {FICHIER_GRID} ({len(df)} lignes)")

    # ------------------------------------------------------------
    # Tableau recapitulatif (pivot) : une ligne par combinaison
    # ------------------------------------------------------------
    pivot = df.pivot_table(
        index=["seuil_ecart", "seuil_pct", "granularite"],
        columns="classe", values="accuracy", aggfunc="first"
    ).reset_index()

    # Reordonner les colonnes
    colonnes = ["seuil_ecart", "seuil_pct", "granularite",
                "Diminution", "Stable", "Augmentation", "Global"]
    colonnes = [c for c in colonnes if c in pivot.columns]
    pivot = pivot[colonnes]

    # Score d'equilibre = min(Diminution, Augmentation)
    if "Diminution" in pivot.columns and "Augmentation" in pivot.columns:
        pivot["score_equilibre"] = pivot[["Diminution", "Augmentation"]].min(axis=1)
    else:
        pivot["score_equilibre"] = None

    # Trier par score d'equilibre decroissant
    pivot = pivot.sort_values("score_equilibre", ascending=False).reset_index(drop=True)

    # Sauvegarder le pivot
    pivot.to_csv(FICHIER_PIVOT, index=False, encoding="utf-8-sig")
    print(f"\nCSV pivot sauvegarde: {FICHIER_PIVOT} ({len(pivot)} lignes)")

    # Affichage du top 15
    print("\n" + "=" * 70)
    print("TOP 15 (par score d'equilibre = min(Diminution, Augmentation))")
    print("=" * 70)
    print(pivot.head(15).to_string(index=False))


# ============================================================
# MAIN
# ============================================================
def main():
    choisis = MODELES_A_LANCER or list(SCRIPTS)

    inconnus = [m for m in choisis if m not in SCRIPTS]
    if inconnus:
        print(f"[!] modele(s) inconnu(s) : {inconnus}. Choix possibles : {list(SCRIPTS)}")
        return

    if MODE == "grid_search":
        mode_grid_search(choisis, SEUILS_ECART, SEUILS_PCT)
    else:
        mode_normal(choisis, SEQUENTIEL)


main()