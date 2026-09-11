# src/prediction/analysis_maps.py
import os
import json
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def generate_prediction_map(data_path=None, output_path=None):
    """
    Génère la carte des prédictions avec curseur temporel.
    """
    
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "predictions_random_forest_densite_2023_2030.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "maps" / "prediction_map" / "predictions_random_forest_densite_2023_2030.html"
    
    df = pd.read_csv(data_path)
    
    TAILLE_GRILLE_DEG = 1.5
    
    df['cell_id'] = (
        "c_" + (df['grille_lat'] * 100).round().astype(int).astype(str)
        + "_" + (df['grille_lon'] * 100).round().astype(int).astype(str)
    )
    
    annees = sorted(df['annee'].unique())
    
    cases_uniques = df.drop_duplicates(subset='cell_id')[['cell_id', 'grille_lat', 'grille_lon']]
    
    cells_info = {
        row['cell_id']: {
            'lat0': row['grille_lat'],
            'lon0': row['grille_lon'],
            'lat1': row['grille_lat'] + TAILLE_GRILLE_DEG,
            'lon1': row['grille_lon'] + TAILLE_GRILLE_DEG,
        }
        for _, row in cases_uniques.iterrows()
    }
    
    def valeur_ou_null(x):
        return None if pd.isna(x) else round(x, 1)
    
    data_par_annee = {}
    for annee in annees:
        data_par_annee[str(int(annee))] = {
            row['cell_id']: {
                'couleur': row['couleur'],
                'categorie': row['categorie'],
                'predit': int(row['predit_affiche']),
                'reel': valeur_ou_null(row['reel_total']),
                'reference': valeur_ou_null(row['reference_totale']),
                'pct': valeur_ou_null(row['pct_evolution']),
            }
            for _, row in df[df['annee'] == annee].iterrows()
        }
    
    lat_centre = df['grille_lat'].mean() + TAILLE_GRILLE_DEG / 2
    lon_centre = df['grille_lon'].mean() + TAILLE_GRILLE_DEG / 2
    
    m = folium.Map(location=[lat_centre, lon_centre], zoom_start=5, tiles='OpenStreetMap')
    nom_carte_js = m.get_name()
    
    title_html = '''
    <h3 align="center" style="font-size:18px; margin-top:5px; font-family: Arial, sans-serif;">
        <b>Tendance des accidents par zone (2023-2030)</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    legend_html = '''
    <div style="position: fixed; bottom: 90px; left: 50px; width: 230px;
                background-color: white; border: 1px solid #ccc; border-radius: 5px;
                padding: 10px; z-index:9999; font-size:12px; font-family: Arial, sans-serif;">
        <b>Tendance vs référence historique</b><br>
        <div><span style="background:#08306b;padding:2px 10px;">&nbsp;</span> Forte diminution</div>
        <div><span style="background:#27ae60;padding:2px 10px;">&nbsp;</span> Faible diminution</div>
        <div><span style="background:#8e44ad;padding:2px 10px;">&nbsp;</span> Stable</div>
        <div><span style="background:#f4d03f;padding:2px 10px;">&nbsp;</span> Faible augmentation</div>
        <div><span style="background:#e74c3c;padding:2px 10px;">&nbsp;</span> Forte augmentation</div>
        <div><span style="background:#cccccc;padding:2px 10px;">&nbsp;</span> Données insuffisantes</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    json_cells_info = json.dumps(cells_info)
    json_data = json.dumps(data_par_annee)
    json_annees = json.dumps([str(int(a)) for a in annees])
    
    js_code = f"""
    <script>
        var cellsInfo = {json_cells_info};
        var dataParAnnee = {json_data};
        var listeAnnees = {json_annees};
        var rectangles = {{}};
        var enLecture = false;
        var intervalLecture = null;

        function valeursCase(cellId, annee) {{
            var donneesAnnee = dataParAnnee[annee] || {{}};
            return donneesAnnee[cellId] || {{
                couleur: '#cccccc', categorie: 'Données insuffisantes',
                predit: null, reel: null, reference: null, pct: null
            }};
        }}

        function contenuPopup(cellId, annee) {{
            var v = valeursCase(cellId, annee);
            var ligneReel = (v.reel !== null) ? v.reel : 'pas encore connue';
            return '<b>Zone ' + cellId + '</b><br>' +
                   'Année : ' + annee + '<br>' +
                   'Référence historique : ' + v.reference + '<br>' +
                   'Nombre d\\'accidents réel : ' + ligneReel + '<br>' +
                   'Nombre d\\'accidents prédit : ' + v.predit + '<br>' +
                   'Évolution (prédit vs référence) : ' + v.pct + '%<br>' +
                   'Catégorie : ' + v.categorie;
        }}

        function initialiserCases() {{
            Object.keys(cellsInfo).forEach(function(cellId) {{
                var b = cellsInfo[cellId];
                var v = valeursCase(cellId, listeAnnees[0]);

                var rect = L.rectangle(
                    [[b.lat0, b.lon0], [b.lat1, b.lon1]],
                    {{ color: 'gray', weight: 0.4, fillColor: v.couleur, fillOpacity: 0.7 }}
                ).addTo({nom_carte_js});

                rect.bindPopup(contenuPopup(cellId, listeAnnees[0]));
                rectangles[cellId] = rect;
            }});
        }}

        function majCarte(annee) {{
            Object.keys(rectangles).forEach(function(cellId) {{
                var v = valeursCase(cellId, annee);
                rectangles[cellId].setStyle({{ fillColor: v.couleur, fillOpacity: 0.7 }});
                rectangles[cellId].setPopupContent(contenuPopup(cellId, annee));
            }});
            document.getElementById('sliderValeur').textContent = annee;
            document.getElementById('sliderAnnee').value = annee;
        }}

        function creerCurseur() {{
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
            label.textContent = 'Année :';
            conteneur.appendChild(label);

            var curseur = document.createElement('input');
            curseur.type = 'range';
            curseur.id = 'sliderAnnee';
            curseur.min = listeAnnees[0];
            curseur.max = listeAnnees[listeAnnees.length - 1];
            curseur.value = listeAnnees[0];
            curseur.step = 1;
            curseur.style.width = '250px';
            conteneur.appendChild(curseur);

            var valeurAffichee = document.createElement('span');
            valeurAffichee.id = 'sliderValeur';
            valeurAffichee.style.fontWeight = 'bold';
            valeurAffichee.style.fontSize = '16px';
            valeurAffichee.style.minWidth = '45px';
            valeurAffichee.style.color = '#8e44ad';
            valeurAffichee.textContent = listeAnnees[0];
            conteneur.appendChild(valeurAffichee);

            var boutonLecture = document.createElement('button');
            boutonLecture.textContent = '▶ Lecture';
            boutonLecture.style.padding = '6px 14px';
            boutonLecture.style.border = 'none';
            boutonLecture.style.borderRadius = '5px';
            boutonLecture.style.backgroundColor = '#4CAF50';
            boutonLecture.style.color = 'white';
            boutonLecture.style.cursor = 'pointer';
            conteneur.appendChild(boutonLecture);

            document.body.appendChild(conteneur);

            curseur.addEventListener('input', function() {{
                document.getElementById('sliderValeur').textContent = this.value;
            }});
            curseur.addEventListener('change', function() {{
                majCarte(this.value);
            }});

            boutonLecture.addEventListener('click', function() {{
                if (enLecture) {{
                    clearInterval(intervalLecture);
                    enLecture = false;
                    this.textContent = '▶ Lecture';
                    this.style.backgroundColor = '#4CAF50';
                }} else {{
                    enLecture = true;
                    this.textContent = '⏹ Arrêter';
                    this.style.backgroundColor = '#ff6b6b';

                    var index = listeAnnees.indexOf(document.getElementById('sliderAnnee').value);
                    if (index === -1 || index === listeAnnees.length - 1) index = 0;

                    intervalLecture = setInterval(function() {{
                        index = (index + 1) % listeAnnees.length;
                        majCarte(listeAnnees[index]);
                    }}, 1500);
                }}
            }});
        }}

        setTimeout(function() {{
            initialiserCases();
            creerCurseur();
        }}, 500);
    </script>
    """
    m.get_root().html.add_child(folium.Element(js_code))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"\n Carte sauvegardée: {output_path}")
    
    return output_path

def generate_tendance_map(data_path=None, output_path=None):
    """
    Genere une carte des tendances predites par le classificateur (2026-2030).
    Meme format que la carte des predictions avec curseur temporel.
    """
    
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "predictions_tendance_2023_2030.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "maps" / "prediction_map" / "tendance_map.html"
    

    
    # 1. Charger les predictions de tendance
    df_tendance = pd.read_csv(data_path)
    
    # 2. Charger les coordonnees depuis le fichier de predictions
    df_coords = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "predictions_random_forest_densite_2023_2030.csv")
    
    # 3. Extraire les coordonnees par zone (une seule fois)
    coords_par_zone = df_coords[['grille_lat', 'grille_lon']].drop_duplicates().reset_index(drop=True)
    coords_par_zone['zone'] = "z_" + coords_par_zone['grille_lat'].astype(str) + "_" + coords_par_zone['grille_lon'].astype(str)    
    # 4. Fusionner les predictions avec les coordonnees
    df_merged = df_tendance.merge(coords_par_zone, on='zone', how='left')
    df_merged = df_merged.dropna(subset=['grille_lat', 'grille_lon'])
    
    # 5. Preparer les donnees pour la carte
    TAILLE_GRILLE_DEG = 1.5
    
    # Identifiant unique par cellule
    df_merged['cell_id'] = (
        "t_" + (df_merged['grille_lat'] * 100).round().astype(int).astype(str)
        + "_" + (df_merged['grille_lon'] * 100).round().astype(int).astype(str)
    )
    
    annees = sorted(df_merged['annee'].unique())
    print(f"Annees: {annees}")
    
    # 6. Bornes des cases
    cases_uniques = df_merged.drop_duplicates(subset='cell_id')[['cell_id', 'grille_lat', 'grille_lon']]
    
    cells_info = {
        row['cell_id']: {
            'lat0': row['grille_lat'],
            'lon0': row['grille_lon'],
            'lat1': row['grille_lat'] + TAILLE_GRILLE_DEG,
            'lon1': row['grille_lon'] + TAILLE_GRILLE_DEG,
        }
        for _, row in cases_uniques.iterrows()
    }
    
    # 7. Donnees par annee (couleur + tendance)
    couleurs_tendance = {
        'Forte diminution': '#08306b',
        'Faible diminution': '#27ae60',
        'Stable': '#8e44ad',
        'Faible augmentation': '#f4d03f',
        'Forte augmentation': '#e74c3c'
    }
    
    data_par_annee = {}
    for annee in annees:
        data_par_annee[str(int(annee))] = {
            row['cell_id']: {
                'couleur': couleurs_tendance.get(row['tendance'], '#cccccc'),
                'tendance': row['tendance']
            }
            for _, row in df_merged[df_merged['annee'] == annee].iterrows()
        }
    
    # 8. Carte de base
    lat_centre = df_merged['grille_lat'].mean() + TAILLE_GRILLE_DEG / 2
    lon_centre = df_merged['grille_lon'].mean() + TAILLE_GRILLE_DEG / 2
    
    m = folium.Map(location=[lat_centre, lon_centre], zoom_start=5, tiles='OpenStreetMap')
    nom_carte_js = m.get_name()
    
    # Titre
    title_html = '''
    <h3 align="center" style="font-size:18px; margin-top:5px; font-family: Arial, sans-serif;">
        <b>Prediction de tendance des accidents (2026-2030)</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Legende
    legend_html = '''
    <div style="position: fixed; bottom: 90px; left: 50px; width: 230px;
                background-color: white; border: 1px solid #ccc; border-radius: 5px;
                padding: 10px; z-index:9999; font-size:12px; font-family: Arial, sans-serif;">
        <b>Tendance predite</b><br>
        <div><span style="background:#08306b;padding:2px 10px;">&nbsp;</span> Forte diminution</div>
        <div><span style="background:#27ae60;padding:2px 10px;">&nbsp;</span> Faible diminution</div>
        <div><span style="background:#8e44ad;padding:2px 10px;">&nbsp;</span> Stable</div>
        <div><span style="background:#f4d03f;padding:2px 10px;">&nbsp;</span> Faible augmentation</div>
        <div><span style="background:#e74c3c;padding:2px 10px;">&nbsp;</span> Forte augmentation</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 9. Code JS (identique a generate_prediction_map)
    json_cells_info = json.dumps(cells_info)
    json_data = json.dumps(data_par_annee)
    json_annees = json.dumps([str(int(a)) for a in annees])
    
    js_code = f"""
    <script>
        var cellsInfo = {json_cells_info};
        var dataParAnnee = {json_data};
        var listeAnnees = {json_annees};
        var rectangles = {{}};
        var enLecture = false;
        var intervalLecture = null;

        function valeursCase(cellId, annee) {{
            var donneesAnnee = dataParAnnee[annee] || {{}};
            return donneesAnnee[cellId] || {{
                couleur: '#cccccc',
                tendance: 'Donnees insuffisantes',
                classe: -1
            }};
        }}

        function contenuPopup(cellId, annee) {{
            var v = valeursCase(cellId, annee);
            return '<b>Zone ' + cellId + '</b><br>' +
                   'Annee : ' + annee + '<br>' +
                   'Tendance : ' + v.tendance + '<br>' +
                   'Classe : ' + v.classe;
        }}

        function initialiserCases() {{
            Object.keys(cellsInfo).forEach(function(cellId) {{
                var b = cellsInfo[cellId];
                var v = valeursCase(cellId, listeAnnees[0]);

                var rect = L.rectangle(
                    [[b.lat0, b.lon0], [b.lat1, b.lon1]],
                    {{ color: 'gray', weight: 0.4, fillColor: v.couleur, fillOpacity: 0.7 }}
                ).addTo({nom_carte_js});

                rect.bindPopup(contenuPopup(cellId, listeAnnees[0]));
                rectangles[cellId] = rect;
            }});
        }}

        function majCarte(annee) {{
            Object.keys(rectangles).forEach(function(cellId) {{
                var v = valeursCase(cellId, annee);
                rectangles[cellId].setStyle({{ fillColor: v.couleur, fillOpacity: 0.7 }});
                rectangles[cellId].setPopupContent(contenuPopup(cellId, annee));
            }});
            document.getElementById('sliderValeur').textContent = annee;
            document.getElementById('sliderAnnee').value = annee;
        }}

        function creerCurseur() {{
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
            label.textContent = 'Annee :';
            conteneur.appendChild(label);

            var curseur = document.createElement('input');
            curseur.type = 'range';
            curseur.id = 'sliderAnnee';
            curseur.min = listeAnnees[0];
            curseur.max = listeAnnees[listeAnnees.length - 1];
            curseur.value = listeAnnees[0];
            curseur.step = 1;
            curseur.style.width = '250px';
            conteneur.appendChild(curseur);

            var valeurAffichee = document.createElement('span');
            valeurAffichee.id = 'sliderValeur';
            valeurAffichee.style.fontWeight = 'bold';
            valeurAffichee.style.fontSize = '16px';
            valeurAffichee.style.minWidth = '45px';
            valeurAffichee.style.color = '#8e44ad';
            valeurAffichee.textContent = listeAnnees[0];
            conteneur.appendChild(valeurAffichee);

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

            curseur.addEventListener('input', function() {{
                document.getElementById('sliderValeur').textContent = this.value;
            }});
            curseur.addEventListener('change', function() {{
                majCarte(this.value);
            }});

            boutonLecture.addEventListener('click', function() {{
                if (enLecture) {{
                    clearInterval(intervalLecture);
                    enLecture = false;
                    this.textContent = 'Lecture';
                    this.style.backgroundColor = '#4CAF50';
                }} else {{
                    enLecture = true;
                    this.textContent = 'Arreter';
                    this.style.backgroundColor = '#ff6b6b';

                    var index = listeAnnees.indexOf(document.getElementById('sliderAnnee').value);
                    if (index === -1 || index === listeAnnees.length - 1) index = 0;

                    intervalLecture = setInterval(function() {{
                        index = (index + 1) % listeAnnees.length;
                        majCarte(listeAnnees[index]);
                    }}, 1500);
                }}
            }});
        }}

        setTimeout(function() {{
            initialiserCases();
            creerCurseur();
        }}, 500);
    </script>
    """
    m.get_root().html.add_child(folium.Element(js_code))
    
    # 10. Sauvegarde
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Carte sauvegardee: {output_path}")
    
    return output_path

def generate_tendance_map_mensuelle(data_path=None, output_path=None):
    """
    Carte des tendances prédites MOIS PAR MOIS (2023-01 à 2030-12), avec curseur
    temporel. Adaptée de generate_tendance_map() : les coordonnées sont extraites
    directement de l'identifiant de zone (zone = "z_{lat}_{lon}") au lieu d'être
    fusionnées depuis un autre fichier -> plus robuste, pas de risque de mismatch
    de format entre deux pipelines différents.
    """
 
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "predictions_tendance_mensuelle_2023_2030.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "maps" / "prediction_map" / "tendance_map_mensuelle.html"

 
    df = pd.read_csv(data_path)
 
    coords = df['zone'].str.split('_', n=2, expand=True)
    df['grille_lat'] = coords[1].astype(float)
    df['grille_lon'] = coords[2].astype(float)
 
    TAILLE_GRILLE_DEG = 1.5
 
    df['cell_id'] = (
        "c_" + (df['grille_lat'] * 100).round().astype(int).astype(str)
        + "_" + (df['grille_lon'] * 100).round().astype(int).astype(str)
    )
 
    liste_mois = sorted(df['mois'].unique())
 
    cases_uniques = df.drop_duplicates(subset='cell_id')[['cell_id', 'grille_lat', 'grille_lon']]
    cells_info = {
        row['cell_id']: {
            'lat0': row['grille_lat'],
            'lon0': row['grille_lon'],
            'lat1': row['grille_lat'] + TAILLE_GRILLE_DEG,
            'lon1': row['grille_lon'] + TAILLE_GRILLE_DEG,
        }
        for _, row in cases_uniques.iterrows()
    }
 
    couleurs_tendance = {
        'Forte diminution': '#08306b',
        'Faible diminution': '#27ae60',
        'Stable': '#8e44ad',
        'Faible augmentation': '#f4d03f',
        'Forte augmentation': '#e74c3c',
        'Données insuffisantes': '#cccccc',
    }
 
    data_par_mois = {}
    for mois in liste_mois:
        data_par_mois[mois] = {
            row['cell_id']: {
                'couleur': couleurs_tendance.get(row['tendance'], '#cccccc'),
                'tendance': row['tendance'],
                'confiance': round(row['confiance'], 1) if 'confiance' in df.columns else None,
            }
            for _, row in df[df['mois'] == mois].iterrows()
        }
 
    lat_centre = df['grille_lat'].mean() + TAILLE_GRILLE_DEG / 2
    lon_centre = df['grille_lon'].mean() + TAILLE_GRILLE_DEG / 2
 
    m = folium.Map(location=[lat_centre, lon_centre], zoom_start=5, tiles='CartoDB Positron')
    nom_carte_js = m.get_name()
 
    title_html = '''
    <h3 align="center" style="font-size:18px; margin-top:5px; font-family: Arial, sans-serif;">
        <b>Prédiction de tendance des accidents, mois par mois (2023-2030)</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
 
    legend_html = '''
    <div style="position: fixed; bottom: 90px; left: 50px; width: 230px;
                background-color: white; border: 1px solid #ccc; border-radius: 5px;
                padding: 10px; z-index:9999; font-size:12px; font-family: Arial, sans-serif;">
        <b>Tendance prédite</b><br>
        <div><span style="background:#08306b;padding:2px 10px;">&nbsp;</span> Forte diminution</div>
        <div><span style="background:#27ae60;padding:2px 10px;">&nbsp;</span> Faible diminution</div>
        <div><span style="background:#8e44ad;padding:2px 10px;">&nbsp;</span> Stable</div>
        <div><span style="background:#f4d03f;padding:2px 10px;">&nbsp;</span> Faible augmentation</div>
        <div><span style="background:#e74c3c;padding:2px 10px;">&nbsp;</span> Forte augmentation</div>
        <div><span style="background:#cccccc;padding:2px 10px;">&nbsp;</span> Données insuffisantes</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
 
    json_cells_info = json.dumps(cells_info)
    json_data = json.dumps(data_par_mois)
    json_liste_mois = json.dumps(liste_mois)
 
    js_code = f"""
    <script>
        var cellsInfo = {json_cells_info};
        var dataParMois = {json_data};
        var listeMois = {json_liste_mois};
        var rectangles = {{}};
        var enLecture = false;
        var intervalLecture = null;
 
        function valeursCase(cellId, mois) {{
            var donneesMois = dataParMois[mois] || {{}};
            return donneesMois[cellId] || {{
                couleur: '#cccccc', tendance: 'Données insuffisantes', confiance: null
            }};
        }}
 
        function contenuPopup(cellId, mois) {{
            var v = valeursCase(cellId, mois);
            var ligneConfiance = (v.confiance !== null) ? v.confiance + '%' : 'N/A';
            return '<b>Zone ' + cellId + '</b><br>' +
                   'Mois : ' + mois + '<br>' +
                   'Tendance : ' + v.tendance + '<br>' +
                   'Confiance : ' + ligneConfiance;
        }}
 
        function initialiserCases() {{
            Object.keys(cellsInfo).forEach(function(cellId) {{
                var b = cellsInfo[cellId];
                var v = valeursCase(cellId, listeMois[0]);
 
                var rect = L.rectangle(
                    [[b.lat0, b.lon0], [b.lat1, b.lon1]],
                    {{ color: 'gray', weight: 0.4, fillColor: v.couleur, fillOpacity: 0.7 }}
                ).addTo({nom_carte_js});
 
                rect.bindPopup(contenuPopup(cellId, listeMois[0]));
                rectangles[cellId] = rect;
            }});
        }}
 
        function majCarte(index) {{
            var mois = listeMois[index];
            Object.keys(rectangles).forEach(function(cellId) {{
                var v = valeursCase(cellId, mois);
                rectangles[cellId].setStyle({{ fillColor: v.couleur, fillOpacity: 0.7 }});
                rectangles[cellId].setPopupContent(contenuPopup(cellId, mois));
            }});
            document.getElementById('sliderValeur').textContent = mois;
            document.getElementById('sliderIndex').value = index;
        }}
 
        function creerCurseur() {{
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
            curseur.style.width = '300px';
            conteneur.appendChild(curseur);
 
            var valeurAffichee = document.createElement('span');
            valeurAffichee.id = 'sliderValeur';
            valeurAffichee.style.fontWeight = 'bold';
            valeurAffichee.style.fontSize = '16px';
            valeurAffichee.style.minWidth = '65px';
            valeurAffichee.style.color = '#8e44ad';
            valeurAffichee.textContent = listeMois[0];
            conteneur.appendChild(valeurAffichee);
 
            var boutonLecture = document.createElement('button');
            boutonLecture.textContent = '▶ Lecture';
            boutonLecture.style.padding = '6px 14px';
            boutonLecture.style.border = 'none';
            boutonLecture.style.borderRadius = '5px';
            boutonLecture.style.backgroundColor = '#4CAF50';
            boutonLecture.style.color = 'white';
            boutonLecture.style.cursor = 'pointer';
            conteneur.appendChild(boutonLecture);
 
            document.body.appendChild(conteneur);
 
            curseur.addEventListener('input', function() {{
                document.getElementById('sliderValeur').textContent = listeMois[this.value];
            }});
            curseur.addEventListener('change', function() {{
                majCarte(parseInt(this.value));
            }});
 
            boutonLecture.addEventListener('click', function() {{
                if (enLecture) {{
                    clearInterval(intervalLecture);
                    enLecture = false;
                    this.textContent = '▶ Lecture';
                    this.style.backgroundColor = '#4CAF50';
                }} else {{
                    enLecture = true;
                    this.textContent = '⏹ Arrêter';
                    this.style.backgroundColor = '#ff6b6b';
 
                    var index = parseInt(document.getElementById('sliderIndex').value);
                    if (index >= listeMois.length - 1) index = -1;
 
                    intervalLecture = setInterval(function() {{
                        index = (index + 1) % listeMois.length;
                        majCarte(index);
                    }}, 400);  // plus rapide que le curseur annuel (96 pas au lieu de 8)
                }}
            }});
        }}
 
        setTimeout(function() {{
            initialiserCases();
            creerCurseur();
        }}, 500);
    </script>
    """
    m.get_root().html.add_child(folium.Element(js_code))
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"\nCarte sauvegardée: {output_path}")
 
    return output_path

def generate_tendance_map_semestrielle(data_path=None, output_path=None):
    """
    Carte des tendances predites par SEMESTRE.
    """

    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "predictions_tendance_semestrielle_2023_2030.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "maps" / "prediction_map" / "tendance_map_semestrielle.html"

    print("=" * 60)
    print("CARTE DES TENDANCES SEMESTRIELLES")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"{len(df)} lignes chargees")

    coords = df['zone'].str.split('_', n=2, expand=True)
    df['grille_lat'] = coords[1].astype(float)
    df['grille_lon'] = coords[2].astype(float)

    TAILLE_GRILLE_DEG = 1.5

    df['cell_id'] = (
        "c_" + (df['grille_lat'] * 100).round().astype(int).astype(str)
        + "_" + (df['grille_lon'] * 100).round().astype(int).astype(str)
    )

    liste_semestres = sorted(df['semestre_annee'].unique())
    print(f"Semestres: {liste_semestres}")

    cases_uniques = df.drop_duplicates(subset='cell_id')[['cell_id', 'grille_lat', 'grille_lon']]
    cells_info = {
        row['cell_id']: {
            'lat0': row['grille_lat'],
            'lon0': row['grille_lon'],
            'lat1': row['grille_lat'] + TAILLE_GRILLE_DEG,
            'lon1': row['grille_lon'] + TAILLE_GRILLE_DEG,
        }
        for _, row in cases_uniques.iterrows()
    }

    couleurs_tendance = {
        'Forte diminution': '#08306b',
        'Faible diminution': '#27ae60',
        'Stable': '#8e44ad',
        'Faible augmentation': '#f4d03f',
        'Forte augmentation': '#e74c3c',
        'Donnees insuffisantes': '#cccccc',
    }

    data_par_semestre = {}
    for sem in liste_semestres:
        data_par_semestre[sem] = {
            row['cell_id']: {
                'couleur': couleurs_tendance.get(row['tendance'], '#cccccc'),
                'tendance': row['tendance'],
                'confiance': round(row['confiance'], 1) if 'confiance' in df.columns else None,
            }
            for _, row in df[df['semestre_annee'] == sem].iterrows()
        }

    lat_centre = df['grille_lat'].mean() + TAILLE_GRILLE_DEG / 2
    lon_centre = df['grille_lon'].mean() + TAILLE_GRILLE_DEG / 2

    m = folium.Map(location=[lat_centre, lon_centre], zoom_start=5, tiles='CartoDB Positron')
    nom_carte_js = m.get_name()

    title_html = '''
    <h3 align="center" style="font-size:18px; margin-top:5px; font-family: Arial, sans-serif;">
        <b>Prediction de tendance des accidents, par semestre (2023-2030)</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    legend_html = '''
    <div style="position: fixed; bottom: 90px; left: 50px; width: 230px;
                background-color: white; border: 1px solid #ccc; border-radius: 5px;
                padding: 10px; z-index:9999; font-size:12px; font-family: Arial, sans-serif;">
        <b>Tendance predite</b><br>
        <div><span style="background:#08306b;padding:2px 10px;">&nbsp;</span> Forte diminution</div>
        <div><span style="background:#27ae60;padding:2px 10px;">&nbsp;</span> Faible diminution</div>
        <div><span style="background:#8e44ad;padding:2px 10px;">&nbsp;</span> Stable</div>
        <div><span style="background:#f4d03f;padding:2px 10px;">&nbsp;</span> Faible augmentation</div>
        <div><span style="background:#e74c3c;padding:2px 10px;">&nbsp;</span> Forte augmentation</div>
        <div><span style="background:#cccccc;padding:2px 10px;">&nbsp;</span> Donnees insuffisantes</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    json_cells_info = json.dumps(cells_info)
    json_data = json.dumps(data_par_semestre)
    json_liste_semestres = json.dumps(liste_semestres)

    js_code = f"""
    <script>
        var cellsInfo = {json_cells_info};
        var dataParSemestre = {json_data};
        var listeSemestres = {json_liste_semestres};
        var rectangles = {{}};
        var enLecture = false;
        var intervalLecture = null;

        function valeursCase(cellId, semestre) {{
            var donneesSemestre = dataParSemestre[semestre] || {{}};
            return donneesSemestre[cellId] || {{
                couleur: '#cccccc', tendance: 'Donnees insuffisantes', confiance: null
            }};
        }}

        function contenuPopup(cellId, semestre) {{
            var v = valeursCase(cellId, semestre);
            var ligneConfiance = (v.confiance !== null) ? v.confiance + '%' : 'N/A';
            return '<b>Zone ' + cellId + '</b><br>' +
                   'Semestre : ' + semestre + '<br>' +
                   'Tendance : ' + v.tendance + '<br>' +
                   'Confiance : ' + ligneConfiance;
        }}

        function initialiserCases() {{
            Object.keys(cellsInfo).forEach(function(cellId) {{
                var b = cellsInfo[cellId];
                var v = valeursCase(cellId, listeSemestres[0]);

                var rect = L.rectangle(
                    [[b.lat0, b.lon0], [b.lat1, b.lon1]],
                    {{ color: 'gray', weight: 0.4, fillColor: v.couleur, fillOpacity: 0.7 }}
                ).addTo({nom_carte_js});

                rect.bindPopup(contenuPopup(cellId, listeSemestres[0]));
                rectangles[cellId] = rect;
            }});
        }}

        function majCarte(index) {{
            var semestre = listeSemestres[index];
            Object.keys(rectangles).forEach(function(cellId) {{
                var v = valeursCase(cellId, semestre);
                rectangles[cellId].setStyle({{ fillColor: v.couleur, fillOpacity: 0.7 }});
                rectangles[cellId].setPopupContent(contenuPopup(cellId, semestre));
            }});
            document.getElementById('sliderValeur').textContent = semestre;
            document.getElementById('sliderIndex').value = index;
        }}

        function creerCurseur() {{
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
            label.textContent = 'Semestre :';
            conteneur.appendChild(label);

            var curseur = document.createElement('input');
            curseur.type = 'range';
            curseur.id = 'sliderIndex';
            curseur.min = 0;
            curseur.max = listeSemestres.length - 1;
            curseur.value = 0;
            curseur.step = 1;
            curseur.style.width = '300px';
            conteneur.appendChild(curseur);

            var valeurAffichee = document.createElement('span');
            valeurAffichee.id = 'sliderValeur';
            valeurAffichee.style.fontWeight = 'bold';
            valeurAffichee.style.fontSize = '16px';
            valeurAffichee.style.minWidth = '65px';
            valeurAffichee.style.color = '#8e44ad';
            valeurAffichee.textContent = listeSemestres[0];
            conteneur.appendChild(valeurAffichee);

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

            curseur.addEventListener('input', function() {{
                document.getElementById('sliderValeur').textContent = listeSemestres[this.value];
            }});
            curseur.addEventListener('change', function() {{
                majCarte(parseInt(this.value));
            }});

            boutonLecture.addEventListener('click', function() {{
                if (enLecture) {{
                    clearInterval(intervalLecture);
                    enLecture = false;
                    this.textContent = 'Lecture';
                    this.style.backgroundColor = '#4CAF50';
                }} else {{
                    enLecture = true;
                    this.textContent = 'Arreter';
                    this.style.backgroundColor = '#ff6b6b';

                    var index = parseInt(document.getElementById('sliderIndex').value);
                    if (index >= listeSemestres.length - 1) index = -1;

                    intervalLecture = setInterval(function() {{
                        index = (index + 1) % listeSemestres.length;
                        majCarte(index);
                    }}, 500);
                }}
            }});
        }}

        setTimeout(function() {{
            initialiserCases();
            creerCurseur();
        }}, 500);
    </script>
    """

    m.get_root().html.add_child(folium.Element(js_code))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Carte sauvegardee: {output_path}")

    return output_path
    
# 1. Carte des prédictions
# generate_prediction_map()
# 2. Carte des tendances 
# generate_tendance_map()
# generate_tendance_map_mensuelle()
generate_tendance_map_semestrielle()
# CARTE DES PRÉDICTIONS (curseur) :
# Référence = 2011-2022 (passé)
# Prédiction = 2023-2030 (futur)