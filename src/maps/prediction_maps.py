# src/prediction/analysis_maps.py
import os
import json
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from pathlib import Path

# ============================================================
# CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ============================================================
# FONCTION 1 : CARTE DES PRÉDICTIONS (EXISTANTE)
# ============================================================

def generate_prediction_map(data_path=None, output_path=None):
    """
    Génère la carte des prédictions avec curseur temporel.
    """
    
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "predictions_random_forest_densite_2023_2030.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "maps" / "prediction_map" / "predictions_random_forest_densite_2023_2030.html"
    
    print("="*60)
    print("🗺️ GÉNÉRATION DE LA CARTE DES PRÉDICTIONS")
    print("="*60)
    
    df = pd.read_csv(data_path)
    print(f"{len(df)} lignes chargées")
    
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


# ============================================================
# FONCTION 2 : CARTE DES ERREURS (MOYENNE SUR 2023-2025)
# ============================================================

def generate_error_map(data_path=None, output_path=None):
    """
    Génère une carte des erreurs MOYENNES du modèle sur 2023-2025.
    """
    
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "predictions_random_forest_densite_2023_2030.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "maps" / "prediction_map" / "error_map.html"
    
    print("="*60)
    print("🗺️ CARTE DES ERREURS MOYENNES (2023-2025)")
    print("="*60)
    
    df = pd.read_csv(data_path)
    print(f"{len(df)} lignes chargées")
    
    # Filtrer les années avec valeurs réelles
    df_eval = df[df['reel_total'].notna()].copy()
    print(f"Années évaluables: {df_eval['annee'].unique().tolist()}")
    print(f"{len(df_eval)} lignes évaluables")
    
    #  CRÉER LA COLONNE 'surestime' AVANT LE GROUPBY
    df_eval['surestime'] = df_eval['predit_total'] > df_eval['reel_total']
    
    # Calculer les erreurs
    df_eval['erreur_absolue'] = abs(df_eval['predit_total'] - df_eval['reel_total'])
    df_eval['erreur_pourcentage'] = np.where(
        df_eval['reel_total'] > 0,
        (df_eval['erreur_absolue'] / df_eval['reel_total']) * 100,
        np.nan
    )
    
    #  AGRÉGER PAR ZONE (MOYENNE SUR 2023-2025)
    zone_error = df_eval.groupby(['grille_lat', 'grille_lon']).agg(
        erreur_moyenne=('erreur_absolue', 'mean'),
        erreur_mediane=('erreur_absolue', 'median'),
        erreur_max=('erreur_absolue', 'max'),
        erreur_pct_moyenne=('erreur_pourcentage', 'mean'),
        nb_annees=('annee', 'count'),
        reel_moyen=('reel_total', 'mean'),
        predit_moyen=('predit_total', 'mean'),
        pct_surestime=('surestime', lambda x: (x.sum() / len(x)) * 100)
    ).reset_index()
    
    #  AJOUTER LA CATÉGORIE "DONNÉES INSUFFISANTES"
    def get_categorie_erreur(row):
        # Si le réel moyen est 0 et la prédiction moyenne est 0 → données insuffisantes
        if row['reel_moyen'] < 0.5 and row['predit_moyen'] < 0.5:
            return 'Données insuffisantes'
        
        # Sinon, on classe selon l'erreur
        erreur = row['erreur_moyenne']
        if erreur < 2:
            return ' Très précis'
        elif erreur < 5:
            return ' Précis'
        elif erreur < 10:
            return ' Imprécis'
        else:
            return ' Très imprécis'
    
    zone_error['categorie_erreur'] = zone_error.apply(get_categorie_erreur, axis=1)
    
    couleurs_erreur = {
        ' Très précis': '#27ae60',
        ' Précis': '#2ecc71',
        ' Imprécis': '#f39c12',
        ' Très imprécis': '#e74c3c',
        'Données insuffisantes': '#cccccc'
    }
    
    # Statistiques globales
    nb_insuffisantes = len(zone_error[zone_error['categorie_erreur'] == 'Données insuffisantes'])
    nb_valides = len(zone_error) - nb_insuffisantes
    
    stats = {
        'mae_moyenne': zone_error[zone_error['categorie_erreur'] != 'Données insuffisantes']['erreur_moyenne'].mean() if nb_valides > 0 else 0,
        'nb_zones_valides': nb_valides,
        'nb_zones_insuffisantes': nb_insuffisantes,
        'pct_surestime_moyen': zone_error[zone_error['categorie_erreur'] != 'Données insuffisantes']['pct_surestime'].mean() if nb_valides > 0 else 0
    }
    
    print(f"\nStatistiques:")
    print(f"   Zones évaluables: {stats['nb_zones_valides']}")
    print(f"   Zones ignorées (données insuffisantes): {stats['nb_zones_insuffisantes']}")
    if stats['nb_zones_valides'] > 0:
        print(f"   MAE moyenne: {stats['mae_moyenne']:.2f} accidents")
        print(f"   % de surestimation: {stats['pct_surestime_moyen']:.1f}%")
    
    # Créer la carte
    print("\n Création de la carte...")
    
    center_lat = zone_error['grille_lat'].mean() + 0.75
    center_lon = zone_error['grille_lon'].mean() + 0.75
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Ajouter les rectangles
    for _, row in zone_error.iterrows():
        lat, lon = row['grille_lat'], row['grille_lon']
        categorie = row['categorie_erreur']
        couleur = couleurs_erreur.get(categorie, '#cccccc')
        
        # Popup différent selon la catégorie
        if categorie == 'Données insuffisantes':
            popup = f"""
            <b>Zone:</b> ({lat:.1f}°, {lon:.1f}°)<br>
            <b>Données insuffisantes</b><br>
            <i>Pas assez d'accidents pour évaluer la fiabilité</i>
            """
        else:
            popup = f"""
            <b>Zone:</b> ({lat:.1f}°, {lon:.1f}°)<br>
            <b>Erreur moyenne:</b> {row['erreur_moyenne']:.1f} accidents<br>
            <b>Réel moyen:</b> {row['reel_moyen']:.1f} accidents<br>
            <b>Prédit moyen:</b> {row['predit_moyen']:.1f} accidents<br>
            <b>Surestimation:</b> {row['pct_surestime']:.0f}% du temps<br>
            <b>Catégorie:</b> {categorie}
            """
        
        folium.Rectangle(
            bounds=[[lat, lon], [lat + 1.5, lon + 1.5]],
            color=couleur,
            weight=1,
            fill=True,
            fillColor=couleur,
            fillOpacity=0.7,
            popup=popup
        ).add_to(m)
    
    # Légende
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: white; padding: 10px; border: 2px solid grey; 
                border-radius: 5px; font-size: 12px; max-width: 250px;">
        <b>Fiabilité du modèle</b><br>
        <span style="color: #27ae60;">●</span> Très précis (erreur < 2)<br>
        <span style="color: #2ecc71;">●</span> Précis (erreur 2-5)<br>
        <span style="color: #f39c12;">●</span> Imprécis (erreur 5-10)<br>
        <span style="color: #e74c3c;">●</span> Très imprécis (erreur > 10)<br>
        <span style="color: #cccccc;">●</span> Données insuffisantes<br>
        <i style="font-size:10px; color:gray;">Erreur en nombre d'accidents</i>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    title_html = '''
    <h3 align="center" style="font-size:16px; font-family: Arial;">
        <b>Fiabilité du modèle (2023-2025)</b><br>
        <span style="font-size:12px; color:gray;">Vert = fiable | Gris = données insuffisantes</span>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"\n Carte sauvegardée: {output_path}")
    
    # Top zones avec forte erreur
    if stats['nb_zones_valides'] > 0:
        print("\n TOP 10 ZONES AVEC LA PLUS FORTE ERREUR:")
        top_erreurs = zone_error[zone_error['categorie_erreur'] != 'Données insuffisantes'].nlargest(10, 'erreur_moyenne')
        for _, row in top_erreurs.iterrows():
            print(f"   ({row['grille_lat']:.1f}°, {row['grille_lon']:.1f}°) "
                  f"| Erreur: {row['erreur_moyenne']:.1f} "
                  f"| Réel: {row['reel_moyen']:.1f} → Prédit: {row['predit_moyen']:.1f}")
    
    return stats


# ============================================================
# FONCTION 3 : CARTE DES ZONES À RISQUE (VERSION CORRIGÉE)
# ============================================================

def generate_risk_zones_map(data_path=None, output_path=None):
    """
    Version corrigée : utilise la référence 2011-2025
    et prédit 2026-2030.
    """
    
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "predictions_random_forest_densite_2023_2030.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "maps" / "maritime_accidents_maps" / "risk_zones_map.html"
    
    print("="*60)
    print(" CARTE DES ZONES À RISQUE (2011-2025 vs 2026-2030)")
    print("="*60)
    
    df = pd.read_csv(data_path)
    
    #  1. Calculer la référence 2011-2022 (historique) depuis le fichier original
    # On utilise les données d'accidents originales
    accidents = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "maritime_accidents.csv")
    
    # Référence 2011-2022
    ref_2011_2022 = accidents[accidents['annee'] <= 2022].groupby('annee').size().mean()
    print(f"Référence 2011-2022: {ref_2011_2022:.2f} accidents/an")
    
    #  2. Calculer la référence 2023-2025 (réel connu)
    ref_2023_2025 = df[df['annee'].isin([2023, 2024, 2025])].groupby('annee')['reel_total'].sum().mean()
    print(f"Référence 2023-2025: {ref_2023_2025:.2f} accidents/an")
    
    #  3. Référence totale 2011-2025
    ref_2011_2025 = (ref_2011_2022 + ref_2023_2025) / 2
    print(f"Référence 2011-2025: {ref_2011_2025:.2f} accidents/an")
    
    #  4. Prédictions 2026-2030
    pred_2026_2030 = df[df['annee'] >= 2026]['predit_total'].mean()
    print(f"Prédiction 2026-2030: {pred_2026_2030:.2f} accidents/an")
    
    #  5. Calcul de l'évolution
    evolution_pct = ((pred_2026_2030 - ref_2011_2025) / ref_2011_2025) * 100
    
    print("\nÉVOLUTION GLOBALE:")
    print(f"   {ref_2011_2025:.1f} → {pred_2026_2030:.1f} = {evolution_pct:+.1f}%")
    
    #  6. Classification des zones
    # (Même logique que avant mais avec la nouvelle référence)
    
    # Agrégation par zone pour 2026-2030
    zone_agg = df[df['annee'] >= 2026].groupby(['grille_lat', 'grille_lon']).agg(
        predit_moyen=('predit_total', 'mean'),
        nb_annees=('annee', 'count')
    ).reset_index()
    
    # Ajouter la référence 2011-2025 par zone (approximation)
    # Pour chaque zone, on prend la moyenne de ses valeurs historiques
    zone_ref = df[df['annee'] <= 2025].groupby(['grille_lat', 'grille_lon']).agg(
        reference_moyenne=('reference_totale', 'mean')
    ).reset_index()
    
    zone_agg = zone_agg.merge(zone_ref, on=['grille_lat', 'grille_lon'], how='left')
    
    # Calcul de l'évolution
    zone_agg['evolution_pct'] = np.where(
        zone_agg['reference_moyenne'] > 0,
        ((zone_agg['predit_moyen'] - zone_agg['reference_moyenne']) / zone_agg['reference_moyenne']) * 100,
        np.nan
    )
    
    zone_agg['ecart_absolu'] = zone_agg['predit_moyen'] - zone_agg['reference_moyenne']
    
    # Classification
    def niveau_risque(row):
        if row['reference_moyenne'] < 0.5:
            return 'Données insuffisantes'
        
        if abs(row['ecart_absolu']) < 1.0:
            return ' Stable'
        
        pct = row['evolution_pct']
        if pct > 30:
            return ' Risque TRÈS ÉLEVÉ'
        elif pct > 15:
            return ' Risque ÉLEVÉ'
        elif pct > 5:
            return ' Risque MODÉRÉ'
        elif pct > -5:
            return ' Stable'
        elif pct > -15:
            return ' Risque FAIBLE'
        else:
            return ' Risque TRÈS FAIBLE'
    
    zone_agg['niveau_risque'] = zone_agg.apply(niveau_risque, axis=1)
    
    # Distribution
    distribution = zone_agg['niveau_risque'].value_counts()
    for niveau, count in distribution.items():
        print(f"   {niveau}: {count} zones ({count/len(zone_agg)*100:.1f}%)")
    
    # Créer la carte    
    center_lat = zone_agg['grille_lat'].mean() + 0.75
    center_lon = zone_agg['grille_lon'].mean() + 0.75
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    couleurs_risque = {
        ' Risque TRÈS ÉLEVÉ': '#8B0000',
        ' Risque ÉLEVÉ': '#FF0000',
        ' Risque MODÉRÉ': '#FF8C00',
        ' Stable': '#8e44ad',
        ' Risque FAIBLE': '#27ae60',
        ' Risque TRÈS FAIBLE': '#08306b',
        'Données insuffisantes': '#cccccc'
    }
    
    for _, row in zone_agg.iterrows():
        lat, lon = row['grille_lat'], row['grille_lon']
        couleur = couleurs_risque.get(row['niveau_risque'], '#cccccc')
        
        folium.Rectangle(
            bounds=[[lat, lon], [lat + 1.5, lon + 1.5]],
            color=couleur,
            weight=1,
            fill=True,
            fillColor=couleur,
            fillOpacity=0.7,
            popup=f"""
            <b>Zone:</b> ({lat:.1f}°, {lon:.1f}°)<br>
            <b>Evolution:</b> {row['evolution_pct']:.1f}%<br>
            <b>Écart absolu:</b> {row['ecart_absolu']:+.1f} accidents<br>
            <b>Référence:</b> {row['reference_moyenne']:.1f} accidents/an<br>
            <b>Prédit:</b> {row['predit_moyen']:.1f} accidents/an<br>
            <b>Niveau:</b> {row['niveau_risque']}
            """
        ).add_to(m)
    
    # Légende
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: white; padding: 10px; border: 2px solid grey; 
                border-radius: 5px; font-size: 12px;">
        <b>Niveau de risque</b><br>
        <span style="color: #8B0000;">●</span> Très Élevé (>30%)<br>
        <span style="color: #FF0000;">●</span> Élevé (15-30%)<br>
        <span style="color: #FF8C00;">●</span> Modéré (5-15%)<br>
        <span style="color: #8e44ad;">●</span> Stable (-5% à +5%)<br>
        <span style="color: #27ae60;">●</span> Faible (-15% à -5%)<br>
        <span style="color: #08306b;">●</span> Très Faible (<-15%)<br>
        <span style="color: #cccccc;">●</span> Données insuffisantes
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    title_html = '''
    <h3 align="center" style="font-size:16px; font-family: Arial;">
        <b> Zones à risque d'accidents (projection 2023-2030)</b><br>
        <span style="font-size:12px; color:gray;">Rouge = augmentation | Vert = diminution | Violet = stable</span>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    return {
        'ref_2011_2025': ref_2011_2025,
        'pred_2026_2030': pred_2026_2030,
        'evolution_globale': evolution_pct,
        'distribution': distribution.to_dict()
    }
    
def generate_tendance_map(data_path=None, output_path=None):
    """
    Genere une carte des tendances predites par le classificateur (2026-2030).
    Meme format que la carte des predictions avec curseur temporel.
    """
    
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "predictions_tendance_2023_2030.csv"
    if output_path is None:
        output_path = PROJECT_ROOT / "maps" / "prediction_map" / "tendance_map.html"
    
    print("="*60)
    print("CARTE DES TENDANCES PREDITES (2026-2030)")
    print("="*60)
    
    # 1. Charger les predictions de tendance
    df_tendance = pd.read_csv(data_path)
    print(f"{len(df_tendance)} lignes chargees")
    
    # 2. Charger les coordonnees depuis le fichier de predictions
    df_coords = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "predictions_random_forest_densite_2023_2030.csv")
    
    # 3. Extraire les coordonnees par zone (une seule fois)
    coords_par_zone = df_coords[['grille_lat', 'grille_lon']].drop_duplicates().reset_index(drop=True)
    coords_par_zone['zone'] = "z_" + coords_par_zone['grille_lat'].astype(str) + "_" + coords_par_zone['grille_lon'].astype(str)    
    # 4. Fusionner les predictions avec les coordonnees
    df_merged = df_tendance.merge(coords_par_zone, on='zone', how='left')
    df_merged = df_merged.dropna(subset=['grille_lat', 'grille_lon'])
    
    print(f"{len(df_merged)} lignes avec coordonnees")
    
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
    
# 1. Carte des prédictions
# print("\n1 Carte des prédictions...")
# generate_prediction_map()

# 2. Carte des erreurs (moyenne 2023-2025)
# print("\n2 Carte des erreurs (moyenne)...")
# error_stats = generate_error_map()

# 3. Carte des zones à risque 
# print("\n3 Carte des zones à risque...")
# risk_stats = generate_risk_zones_map()

# 4. Carte des tendances (NOUVELLE)
print("4. Carte des tendances...")
generate_tendance_map()

# CARTE DES PRÉDICTIONS (curseur) :
# Référence = 2011-2022 (passé)
# Prédiction = 2023-2030 (futur)
# On compare futur vs passé

# CARTE DES ERREURS :
# Référence = Réel 2023-2025
# Prédiction = Prédit 2023-2025
# On valide le modèle

# CARTE DES RISQUES (nouvelle) :
# Référence = 2011-2025 (tout)
# Prédiction = 2026-2030 (futur lointain)
# On voit la tendance long terme