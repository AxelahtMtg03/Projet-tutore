# src/prediction/maps_mensuel_3classes.py
"""
Cartes du modele mensuel a 3 classes (Diminution / Stable / Augmentation),
en cellules de 1,5 x 1,5 degres (memes zones que le modele).

  1. generate_prediction_map_3classes : tendance predite mois par mois (curseur).
     La legende compte, pour le mois affiche, les zones predites par classe et le reel
     (le reel n'apparait que sur la periode de test, jamais sur les vrais mois futurs).
  2. generate_error_map : taux de bonnes predictions par zone sur 2023-2025
     (rouge = 0 %, vert = 100 %), avec "Predit : ..." et "Reel : ..." dans le popup.

Une seule configuration est lue (SEUIL_ECART / SEUIL_PCT, la meilleure du grid search) :
  carte 1 <- predictions_tendance_mensuelle_subdivision_e{ecart}_p{pct}.csv  (toutes les predictions)
             + comparaison_..._e{ecart}_p{pct}.csv  (uniquement pour le reel de 2023-2025)
  carte 2 <- comparaison_tendance_mensuelle_subdivision_e{ecart}_p{pct}.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MAPS_PATH = PROJECT_ROOT / "maps" / "prediction_map"

TAILLE_GRILLE_DEG = 1.5
# Si le fond de carte ne s'affiche pas (les versions recentes de folium demandent une cle pour
# CartoDB), mets "OpenStreetMap".
FOND_DE_CARTE = "OpenStreetMap"
SEUIL_ECART = 1.5     # seuils du modele retenu
SEUIL_PCT = 20.0

CLASSES = ["Diminution", "Stable", "Augmentation"]
COULEURS = ["#2980b9", "#8e44ad", "#e74c3c"]      # Diminution = bleu, Stable = violet, Augmentation = rouge
ANNEE_FIN_TEST = 2025


# ============================================================
# OUTILS COMMUNS
# ============================================================
def _suffixe(seuil_ecart, seuil_pct):
    return f"_e{float(seuil_ecart)}_p{float(seuil_pct)}"


def _ajouter_cellules(df):
    """Coordonnees du coin bas-gauche de la zone, lues dans l'identifiant 'z_lat_lon'."""
    coords = df["zone"].str.split("_", n=2, expand=True)
    df["grille_lat"] = coords[1].astype(float)
    df["grille_lon"] = coords[2].astype(float)
    df["cell_id"] = (
        "c_" + (df["grille_lat"] * 100).round().astype(int).astype(str)
        + "_" + (df["grille_lon"] * 100).round().astype(int).astype(str)
    )
    return df


def _cells_info(df):
    uniques = df.drop_duplicates(subset="cell_id")[["cell_id", "zone", "grille_lat", "grille_lon"]]
    return {
        r["cell_id"]: {
            "zone": r["zone"],
            "lat0": r["grille_lat"], "lon0": r["grille_lon"],
            "lat1": r["grille_lat"] + TAILLE_GRILLE_DEG,
            "lon1": r["grille_lon"] + TAILLE_GRILLE_DEG,
        }
        for _, r in uniques.iterrows()
    }


def _mois_texte(comp):
    """'YYYY-MM' a partir des colonnes annee + mois du fichier de comparaison."""
    if pd.api.types.is_numeric_dtype(comp["mois"]):
        num = comp["mois"].astype(int)
    else:                                   # deja "YYYY-MM"
        num = comp["mois"].astype(str).str.split("-").str[-1].astype(int)
    return comp["annee"].astype(int).astype(str) + "-" + num.astype(str).str.zfill(2)


def _creer_carte(df):
    lat_centre = df["grille_lat"].mean() + TAILLE_GRILLE_DEG / 2
    lon_centre = df["grille_lon"].mean() + TAILLE_GRILLE_DEG / 2
    return folium.Map(location=[lat_centre, lon_centre], zoom_start=5, tiles=FOND_DE_CARTE)


# ============================================================
# 1. CARTE DES PREDICTIONS (curseur mensuel)
# ============================================================
def generate_prediction_map_3classes(seuil_ecart=SEUIL_ECART, seuil_pct=SEUIL_PCT, output_path=None):
    suf = _suffixe(seuil_ecart, seuil_pct)
    pred = pd.read_csv(DATA_PATH / f"predictions_tendance_mensuelle_subdivision{suf}.csv", encoding="utf-8-sig")
    comp = pd.read_csv(DATA_PATH / f"comparaison_tendance_mensuelle_subdivision{suf}.csv", encoding="utf-8-sig")
    if output_path is None:
        output_path = MAPS_PATH / f"carte_predictions_mensuelle{suf}.html"
    output_path = Path(output_path)

    pred = _ajouter_cellules(pred)
    cells_info = _cells_info(pred)
    liste_mois = sorted(pred["mois"].unique())

    # tendance reelle : seulement les mois de la periode de test qui ont une reference historique
    reel = comp.set_index(["zone", _mois_texte(comp)])["tendance_reelle"].to_dict()

    idx = {c: i for i, c in enumerate(CLASSES)}
    donnees = {}
    for mois, g in pred.groupby("mois"):
        donnees[mois] = {
            r.cell_id: [
                idx[r.tendance], round(r.confiance, 1),
                round(r.proba_Diminution, 1), round(r.proba_Stable, 1), round(r.proba_Augmentation, 1),
                idx[reel[(r.zone, mois)]] if (r.zone, mois) in reel else -1,     # -1 = pas de reel
            ]
            for r in g.itertuples()
        }

    m = _creer_carte(pred)
    nom_carte_js = m.get_name()

    titre = f"""
    <h3 align="center" style="font-size:18px; margin-top:5px; font-family: Arial, sans-serif;">
        <b>Prédiction de tendance des accidents, mois par mois (2023-2030)</b>
    </h3>
    <p align="center" style="font-size:12px; margin:0; font-family: Arial, sans-serif;">
        Modèle mensuel à 3 classes &middot; seuils : écart {seuil_ecart} accident, {seuil_pct} % &middot; zones de {TAILLE_GRILLE_DEG}° &times; {TAILLE_GRILLE_DEG}°
    </p>
    """
    m.get_root().html.add_child(folium.Element(titre))

    legende = """
    <div id="legendeTendance" style="position: fixed; bottom: 90px; left: 50px; width: 250px;
                background-color: white; border: 1px solid #ccc; border-radius: 5px;
                padding: 10px; z-index:9999; font-size:12px; font-family: Arial, sans-serif;"></div>
    """
    m.get_root().html.add_child(folium.Element(legende))

    js = """
    <script>
        var cellsInfo = __CELLS__;
        var donnees = __DONNEES__;
        var listeMois = __MOIS__;
        var finTest = "__FIN_TEST__";
        var CLASSES = __CLASSES__;
        var COULEURS = __COULEURS__;
        var rectangles = {};
        var enLecture = false;
        var intervalLecture = null;

        function valeursCase(cellId, mois) {
            var d = donnees[mois] || {};
            return d[cellId] || null;
        }

        function couleurCase(v) { return v ? COULEURS[v[0]] : '#cccccc'; }

        // Legende : nombre de zones par classe pour le mois affiche.
        // Colonne "réel" uniquement si le réel est connu (periode de test) ; sinon rien.
        function majLegende(mois) {
            var d = donnees[mois] || {};
            var predTous = [0, 0, 0], predCmp = [0, 0, 0], reelCmp = [0, 0, 0], nTous = 0, nCmp = 0;
            Object.keys(d).forEach(function(c) {
                var v = d[c];
                predTous[v[0]]++; nTous++;
                if (v[5] >= 0) { predCmp[v[0]]++; reelCmp[v[5]]++; nCmp++; }
            });
            var comparaison = nCmp > 0;
            var td = 'align="right" style="padding-left:14px;"';
            var h = '<b>Tendance — ' + mois + '</b>' +
                    '<table style="margin-top:6px; border-collapse:collapse;"><tr style="color:#555;"><td></td>' +
                    '<td ' + td + '>prédit</td>' + (comparaison ? '<td ' + td + '>réel</td>' : '') + '</tr>';
            for (var i = 0; i < 3; i++) {
                h += '<tr><td><span style="background:' + COULEURS[i] + ';padding:2px 10px;">&nbsp;</span> ' + CLASSES[i] + '</td>' +
                     '<td ' + td + '>' + (comparaison ? predCmp[i] : predTous[i]) + '</td>' +
                     (comparaison ? '<td ' + td + '>' + reelCmp[i] + '</td>' : '') + '</tr>';
            }
            document.getElementById('legendeTendance').innerHTML = h;
        }

        function contenuPopup(cellId, mois) {
            var v = valeursCase(cellId, mois);
            var txt = '<b>Zone ' + cellsInfo[cellId].zone + '</b><br>Mois : ' + mois + '<br>';
            if (!v) return txt + 'Données insuffisantes';
            txt += 'Tendance prédite : <b>' + CLASSES[v[0]] + '</b> (confiance ' + v[1] + '%)<br>' +
                   'Probabilités : Diminution ' + v[2] + '% · Stable ' + v[3] + '% · Augmentation ' + v[4] + '%';
            if (v[5] >= 0) {   // le reel n'est affiche que s'il existe (jamais sur un vrai mois futur)
                txt += '<br>Tendance réelle : <b>' + CLASSES[v[5]] + '</b> '            }
            return txt;
        }

        function initialiserCases() {
            Object.keys(cellsInfo).forEach(function(cellId) {
                var b = cellsInfo[cellId];
                var v = valeursCase(cellId, listeMois[0]);
                var rect = L.rectangle(
                    [[b.lat0, b.lon0], [b.lat1, b.lon1]],
                    { color: 'gray', weight: 0.4, fillColor: couleurCase(v), fillOpacity: 0.7 }
                ).addTo(__MAP__);
                rect.bindPopup(contenuPopup(cellId, listeMois[0]));
                rectangles[cellId] = rect;
            });
        }

        function majCarte(index) {
            var mois = listeMois[index];
            Object.keys(rectangles).forEach(function(cellId) {
                var v = valeursCase(cellId, mois);
                rectangles[cellId].setStyle({ fillColor: couleurCase(v), fillOpacity: 0.7 });
                rectangles[cellId].setPopupContent(contenuPopup(cellId, mois));
            });
            document.getElementById('sliderValeur').textContent = mois;
            document.getElementById('sliderPhase').textContent =
                (mois <= finTest) ? 'période de test (2023-2025)' : 'prévision';
            document.getElementById('sliderIndex').value = index;
            majLegende(mois);
        }

        function creerCurseur() {
            var conteneur = document.createElement('div');
            conteneur.style.position = 'fixed';
            conteneur.style.bottom = '30px';
            conteneur.style.left = '50%';
            conteneur.style.transform = 'translateX(-50%)';
            conteneur.style.zIndex = '1000';
            conteneur.style.backgroundColor = 'white';
            conteneur.style.padding = '12px 20px';
            conteneur.style.borderRadius = '10px';
            conteneur.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
            conteneur.style.display = 'flex';
            conteneur.style.alignItems = 'center';
            conteneur.style.gap = '12px';
            conteneur.style.fontFamily = 'Arial, sans-serif';

            var label = document.createElement('span');
            label.style.fontWeight = 'bold';
            label.textContent = 'Mois :';
            conteneur.appendChild(label);

            var curseur = document.createElement('input');
            curseur.type = 'range';
            curseur.id = 'sliderIndex';
            curseur.min = 0;
            curseur.max = listeMois.length - 1;
            curseur.value = 0;
            curseur.step = 1;
            curseur.style.width = '320px';
            conteneur.appendChild(curseur);

            var valeurAffichee = document.createElement('span');
            valeurAffichee.id = 'sliderValeur';
            valeurAffichee.style.fontWeight = 'bold';
            valeurAffichee.style.fontSize = '16px';
            valeurAffichee.style.minWidth = '70px';
            valeurAffichee.style.color = '#8e44ad';
            valeurAffichee.textContent = listeMois[0];
            conteneur.appendChild(valeurAffichee);

            var phase = document.createElement('span');
            phase.id = 'sliderPhase';
            phase.style.fontSize = '11px';
            phase.style.color = '#555';
            phase.style.minWidth = '160px';
            phase.textContent = (listeMois[0] <= finTest) ? 'période de test (2023-2025)' : 'prévision';
            conteneur.appendChild(phase);

            var boutonLecture = document.createElement('button');
            boutonLecture.textContent = 'Lecture';
            boutonLecture.style.padding = '6px 14px';
            boutonLecture.style.border = 'none';
            boutonLecture.style.borderRadius = '5px';
            boutonLecture.style.backgroundColor = '#4CAF50';
            boutonLecture.style.color = 'white';
            boutonLecture.style.cursor = 'pointer';
            conteneur.appendChild(boutonLecture);

            document.body.appendChild(conteneur);

            curseur.addEventListener('input', function() {
                document.getElementById('sliderValeur').textContent = listeMois[this.value];
            });
            curseur.addEventListener('change', function() {
                majCarte(parseInt(this.value));
            });

            boutonLecture.addEventListener('click', function() {
                if (enLecture) {
                    clearInterval(intervalLecture);
                    enLecture = false;
                    this.textContent = 'Lecture';
                    this.style.backgroundColor = '#4CAF50';
                } else {
                    enLecture = true;
                    this.textContent = 'Arrêter';
                    this.style.backgroundColor = '#ff6b6b';
                    var index = parseInt(document.getElementById('sliderIndex').value);
                    if (index >= listeMois.length - 1) index = -1;
                    intervalLecture = setInterval(function() {
                        index = (index + 1) % listeMois.length;
                        majCarte(index);
                    }, 400);
                }
            });
        }

        setTimeout(function() {
            initialiserCases();
            creerCurseur();
            majLegende(listeMois[0]);
        }, 500);
    </script>
    """
    js = (js.replace("__CELLS__", json.dumps(cells_info))
            .replace("__DONNEES__", json.dumps(donnees))
            .replace("__MOIS__", json.dumps(liste_mois))
            .replace("__FIN_TEST__", f"{ANNEE_FIN_TEST}-12")
            .replace("__CLASSES__", json.dumps(CLASSES))
            .replace("__COULEURS__", json.dumps(COULEURS))
            .replace("__MAP__", nom_carte_js))
    m.get_root().html.add_child(folium.Element(js))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Carte sauvegardee: {output_path}")
    return output_path


# ============================================================
# 2. CARTE DES ERREURS (par zone, sur 2023-2025)
# ============================================================
def generate_error_map(seuil_ecart=SEUIL_ECART, seuil_pct=SEUIL_PCT, n_min_mois=12, output_path=None):
    """
    Taux de bonnes predictions de chaque zone sur toute la periode de test :
      0 % rouge  |  25 % orange  |  50 % jaune  |  75 % jaune-vert  |  100 % vert
    Les zones evaluees sur moins de n_min_mois mois sont grisees (un taux calcule sur
    2 ou 3 mois ne veut rien dire). Mets n_min_mois=1 pour tout colorer.
    """
    suf = _suffixe(seuil_ecart, seuil_pct)
    comp = pd.read_csv(DATA_PATH / f"comparaison_tendance_mensuelle_subdivision{suf}.csv", encoding="utf-8-sig")
    if output_path is None:
        output_path = MAPS_PATH / f"carte_erreurs_mensuelle{suf}.html"
    output_path = Path(output_path)

    comp = _ajouter_cellules(comp)
    cells_info = _cells_info(comp)
    idx = {c: i for i, c in enumerate(CLASSES)}

    # ---- agregats par zone
    zones = {}
    for cell_id, g in comp.groupby("cell_id"):
        n = int(len(g))
        k = int(g["correct"].sum())
        zones[cell_id] = {
            "n": n, "k": k, "acc": round(k / n * 100, 1),
            "pred": [int((g["tendance"] == c).sum()) for c in CLASSES],          # ce que le modele a predit
            "reel": [int((g["tendance_reelle"] == c).sum()) for c in CLASSES],   # ce qui s'est passe
        }

    # ---- metriques globales de la periode de test
    y_vrai, y_pred = comp["tendance_reelle"], comp["tendance"]
    prec, rappel, _, support = precision_recall_fscore_support(y_vrai, y_pred, labels=CLASSES, zero_division=0)


    # ---- diagnostic : la carte est-elle informative ?
    z_df = pd.DataFrame(zones).T
    z_ok = z_df[z_df["n"] >= n_min_mois]
    print(f"Zones evaluees : {len(z_df)} | colorees (n >= {n_min_mois} mois) : {len(z_ok)} | grisees : {len(z_df) - len(z_ok)}")
    if len(z_ok) >= 5:
        corr = z_ok["acc"].astype(float).corr(z_ok["n"].astype(float), method="spearman")
        print(f"Taux de bonnes predictions par zone : min {z_ok['acc'].min():.0f}% | "
              f"mediane {z_ok['acc'].median():.0f}% | max {z_ok['acc'].max():.0f}%")
        print(f"Correlation (Spearman) taux de reussite / nb de mois evalues : {corr:+.2f}")

    m = _creer_carte(comp)
    nom_carte_js = m.get_name()

    titre = f"""
    <h3 align="center" style="font-size:18px; margin-top:5px; font-family: Arial, sans-serif;">
        <b>Qualité des prédictions par zone, 2023-2025 (mensuel)</b>
    </h3>
    <p align="center" style="font-size:12px; margin:0; font-family: Arial, sans-serif;">
        Seuils : écart {seuil_ecart} accident, {seuil_pct} % &middot; zones de {TAILLE_GRILLE_DEG}° &times; {TAILLE_GRILLE_DEG}°
        &middot; grisé = moins de {n_min_mois} mois évalués
    </p>
    """
    m.get_root().html.add_child(folium.Element(titre))

    legende = """
    <div style="position: fixed; bottom: 40px; left: 50px; width: 260px;
                background-color: white; border: 1px solid #ccc; border-radius: 5px;
                padding: 10px; z-index:9999; font-size:12px; font-family: Arial, sans-serif;">
        <b>Part des mois bien prédits</b>
        <div style="height: 14px; margin-top: 6px; border-radius: 3px;
                    background: linear-gradient(to right, #d73027 0%, #fc8d59 25%, #ffd94a 50%, #a6d96a 75%, #1a9850 100%);"></div>
        <div style="display:flex; justify-content:space-between; margin-top:2px;">
            <span>0 %</span><span>25 %</span><span>50 %</span><span>75 %</span><span>100 %</span>
        </div>
        <div style="margin-top:6px;"><span style="background:#cccccc;padding:2px 10px;">&nbsp;</span> Trop peu de mois évalués</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legende))

    js = """
    <script>
        var cellsInfo = __CELLS__;
        var zones = __ZONES__;
        var CLASSES = __CLASSES__;
        var N_MIN = __NMIN__;
        var rectangles = {};

        // 0 % rouge, 25 % orange, 50 % jaune, 75 % jaune-vert, 100 % vert
        var STOPS = [[0, [215,48,39]], [25, [252,141,89]], [50, [255,217,74]],
                     [75, [166,217,106]], [100, [26,152,80]]];

        function interp(stops, x) {
            if (x <= stops[0][0]) return stops[0][1];
            for (var i = 1; i < stops.length; i++) {
                if (x <= stops[i][0]) {
                    var t = (x - stops[i-1][0]) / (stops[i][0] - stops[i-1][0]);
                    var a = stops[i-1][1], b = stops[i][1];
                    return [0, 1, 2].map(function(k) { return Math.round(a[k] + t * (b[k] - a[k])); });
                }
            }
            return stops[stops.length - 1][1];
        }

        function styleCase(cellId) {
            var z = zones[cellId];
            if (!z || z.n < N_MIN) return { fillColor: '#cccccc', fillOpacity: 0.5, dashArray: '4', color: 'gray', weight: 0.6 };
            var c = interp(STOPS, z.acc);
            return { fillColor: 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')', fillOpacity: 0.75, dashArray: null, color: 'gray', weight: 0.5 };
        }

        function contenuPopup(cellId) {
            var z = zones[cellId];
            var t = '<b>Zone ' + cellsInfo[cellId].zone + '</b><br>';
            if (!z) return t + 'Aucune donnée évaluée';
            function ligne(nom, v) {
                return nom + ' : ' + CLASSES.map(function(c, i) { return c + ' <b>' + v[i] + '</b>'; }).join(' · ') + '<br>';
            }
            return t + z.n + ' mois évalués' + (z.n < N_MIN ? ' <i>(trop peu, non coloré)</i>' : '') + '<br>' +
                   'Bien prédits : <b>' + z.acc + '%</b> (' + z.k + '/' + z.n + ')<br><br>' +
                   ligne('Prédit', z.pred) + ligne('Réel', z.reel);
        }

        function initialiserCases() {
            Object.keys(cellsInfo).forEach(function(cellId) {
                var b = cellsInfo[cellId];
                var rect = L.rectangle([[b.lat0, b.lon0], [b.lat1, b.lon1]], styleCase(cellId)).addTo(__MAP__);
                rect.bindPopup(contenuPopup(cellId), { maxWidth: 320 });
                rectangles[cellId] = rect;
            });
        }

        setTimeout(initialiserCases, 500);
    </script>
    """
    js = (js.replace("__CELLS__", json.dumps(cells_info))
            .replace("__ZONES__", json.dumps(zones))
            .replace("__CLASSES__", json.dumps(CLASSES))
            .replace("__NMIN__", str(int(n_min_mois)))
            .replace("__MAP__", nom_carte_js))
    m.get_root().html.add_child(folium.Element(js))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Carte sauvegardee: {output_path}")
    return output_path


generate_prediction_map_3classes()
generate_error_map()