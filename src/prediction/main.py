"""
Lance les modeles Random Forest (annuel, semestre, trimestre, mois) en meme
temps, affiche leurs prints (prefixes par le nom du modele) et rassemble les
metriques dans un seul CSV de synthese.

Format du CSV de sortie (data/processed/synthese_tendances.csv) :
    granularite,classe,accuracy
    annuel,Diminution,43.2
    annuel,Stable,76.0
    annuel,Augmentation,21.4
    annuel,Global,57.4
    ...

Usage :
    python main.py                    # tous les modeles, en parallele
    python main.py annuel mois        # seulement certains modeles
    python main.py --sequentiel       # un apres l'autre au lieu de en meme temps
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

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

MOTIF_METRIQUES = "metriques_tendance_*.csv"
FICHIER_SYNTHESE = DATA_PATH / "synthese_tendances.csv"

ORDRE_CLASSES = ["Diminution", "Stable", "Augmentation"]
ORDRE_GRANULARITES = ["annuel", "semestriel", "trimestriel", "mensuel"]

_verrou_print = threading.Lock()


def lancer_script(nom, script):
    """Lance un script dans son propre processus et affiche sa sortie en direct."""
    debut = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        cwd=str(script.parent), env=env,
    )
    for ligne in proc.stdout:
        with _verrou_print:
            print(f"[{nom}] {ligne.rstrip()}")
    code = proc.wait()
    return {"modele": nom, "statut": "OK" if code == 0 else f"ERREUR (code {code})",
            "duree_s": round(time.time() - debut, 1)}


def construire_synthese(resume):
    """
    Transforme le CSV long (granularite, section, detail, metrique, valeur, n)
    en tableau condense : granularite, classe, accuracy.

    - section "test" / detail "globale" -> classe = "Global"
    - section "par_tendance"            -> classe = detail (Diminution/Stable/Augmentation)
    """
    masque_global = (
        (resume["section"] == "test")
        & (resume["detail"] == "globale")
        & (resume["metrique"] == "accuracy")
    )
    masque_tendance = (
        (resume["section"] == "par_tendance")
        & (resume["metrique"] == "accuracy")
    )

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
    synthese = (
        synthese.sort_values(["_g", "_c"])
        .drop(columns=["_g", "_c"])
        .reset_index(drop=True)
    )
    return synthese[["granularite", "classe", "accuracy"]]


def main():
    parser = argparse.ArgumentParser(description="Lance les modeles RF de prediction de tendance")
    parser.add_argument("modeles", nargs="*",
                        help=f"modeles a lancer parmi {list(SCRIPTS)} (par defaut : tous)")
    parser.add_argument("--sequentiel", action="store_true",
                        help="lancer les modeles un par un au lieu de en parallele")
    args = parser.parse_args()

    inconnus = [m for m in args.modeles if m not in SCRIPTS]
    if inconnus:
        parser.error(f"modele(s) inconnu(s) : {inconnus}. Choix possibles : {list(SCRIPTS)}")
    choisis = args.modeles or list(SCRIPTS)

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
    mode = "sequentiel" if args.sequentiel else "parallele"
    print(f"LANCEMENT ({mode}) : {', '.join(a_lancer)}")
    print("=" * 70)

    debut_global = time.time()
    if args.sequentiel:
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


main()