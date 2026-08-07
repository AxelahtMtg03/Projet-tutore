import pandas as pd
import folium
import branca.colormap as cm
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson
import json

df = pd.read_csv("data/processed/maritime_accidents.csv")

df = df.dropna(subset=['lat', 'long'])

center_lat = df['lat'].mean()
center_lon = df['long'].mean()

couleurs = {
    'Very serious': 'red', 
    'Serious': 'orange',
    'Marine incident': 'blue',
    'less serious': 'gray'
}

def accidents():
    """Carte 1 : tous les accidents, avec regroupement en clusters et couleur selon la gravité"""
    m1 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>Tous les accidents de bateau</b></h3>
                '''
    m1.get_root().html.add_child(folium.Element(title_html))

    marker_cluster = MarkerCluster().add_to(m1)

    for idx, row in df.iterrows():
        # Choisit la couleur selon la gravité de l'accident
        color = couleurs.get(row['gravite'], 'gray')
        
        #permet de dire si le facteur de l'accident est humain ou pas
        if pd.notna(row['type_accident']):
            cause = row['type_accident']
        elif pd.notna(row['cause_accident_humain']):
            cause = row['cause_accident_humain']
        else:
            cause = "Inconnue"
        popup_text = f"""
        <b>Date:</b> {row['Date of occurrence']}<br>
        <b>Heure:</b> {row['heure']}h<br>
        <b>Gravité:</b> {row['gravite']}<br>
        <b>Port:</b> {row['port']}<br>
        <b>Bateau:</b> {row['bateau']}<br>
        <b>Cause de l'accident:</b> {cause}<br>
        <b>Saison:</b> {row['saison']}<br>
        <b>Année:</b> {row['annee']}
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=5,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(marker_cluster)

    # Ajoute une légende fixe en bas à gauche de la carte
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><b>Légende - Gravité</b></p>
        <p><span style="color: red;">●</span> Very Serious</p>
        <p><span style="color: orange;">●</span> Serious</p>
        <p><span style="color: blue;">●</span> Marine incident</p>
        <p><span style="color: gray;">●</span> less serious</p>
    </div>
    '''
    m1.get_root().html.add_child(folium.Element(legend_html))

    m1.save("maps/maritime_accidents_maps/accident_map.html")
    
def heatmap():
    """Carte 2 : carte de chaleur (densité des accidents)"""
    m2 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    heat_data = [[row['lat'], row['long']] for idx, row in df.iterrows()]

    HeatMap(heat_data, 
            radius=15,
            blur=10,
            max_zoom=1,
            min_opacity=0.3
    ).add_to(m2)

    title_html2 = '''
                <h3 align="center" style="font-size:16px"><b>Carte de chaleur des accidents</b></h3>
                '''
    m2.get_root().html.add_child(folium.Element(title_html2))

    m2.save("maps/maritime_accidents_maps/heatmap_accident.html")

def accident_annee():
    """Carte 3 : accidents par année, avec une couche activable/désactivable par année"""
    m3 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    # Crée un groupe de marqueurs (une couche) par année
    annees = sorted(df['annee'].unique())

    for annee in annees:
        fg = folium.FeatureGroup(name=f'Année {annee}')
        df_annee = df[df['annee'] == annee]
        
        for idx, row in df_annee.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=4,
                popup=f"Date: {row['Date of occurrence']}<br>Gravité: {row['gravite']}",
                color='blue',
                fill=True,
                fillOpacity=0.6
            ).add_to(fg)
        
        fg.add_to(m3)

    # Ajoute le contrôle permettant d'afficher/masquer chaque couche (année)
    folium.LayerControl().add_to(m3)

    title_html3 = '''
                <h3 align="center" style="font-size:16px"><b>Accidents par année</b></h3>
                '''
    m3.get_root().html.add_child(folium.Element(title_html3))

    m3.save("maps/maritime_accidents_maps/carte_accidents_par_annee.html")

def accident_temps_animation():
    """Carte 4 : animation temporelle des accidents avec un curseur de temps"""
    df['date_str'] = pd.to_datetime(df['Date of occurrence']).dt.strftime('%Y-%m-%d %H:%M:%S')

    features = []
    for idx, row in df.iterrows():
        # Détermine la couleur selon la gravité (gris si gravité inconnue ou absente)
        grav = row['gravite'] if row['gravite'] in couleurs else 'Unknown'
        color = couleurs.get(grav, 'gray')
        
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['long'], row['lat']]
            },
            'properties': {
                'time': row['date_str'],
                'popup': f"""
                    Date: {row['Date of occurrence']}<br>
                    Heure: {row['heure']}h<br>
                    Gravité: {row['gravite']}<br>
                    Port: {row['port']}<br>
                    Bateau: {row['bateau']}
                """,
                'style': {
                    'color': color,
                    'radius': 6,
                    'fillColor': color,
                    'fillOpacity': 0.8,
                    'weight': 1
                }
            }
        }
        features.append(feature)

    geojson_data = {
        'type': 'FeatureCollection',
        'features': features
    }

    m4 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    # Ajoute le plugin qui affiche les points progressivement selon la date, avec un curseur temporel
    TimestampedGeoJson(
        geojson_data,
        period='P1M',  # 1 mois par étape (tu peux mettre 'P1D' pour 1 jour)
        duration='P1M',
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=5,
        loop_button=True,
        date_options='YYYY-MM-DD',
        time_slider_drag_update=True
    ).add_to(m4)

    title_html4 = '''
                <h3 align="center" style="font-size:16px"><b>Évolution temporelle des accidents</b></h3>
                <p align="center" style="font-size:12px">Utilise le curseur en bas pour voir l'évolution dans le temps</p>
                '''
    m4.get_root().html.add_child(folium.Element(title_html4))

    m4.save("maps/maritime_accidents_maps/accidents_timeline.html")

def accident_grille(taille_grille=1, rayon_voisinage=0):
    """"""
    m5 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    # Affecte chaque accident à une case de la grille (arrondi vers le bas au multiple de taille_grille)
    df_grille = df.copy()
    df_grille['case_lat'] = (df_grille['lat'] // taille_grille) * taille_grille
    df_grille['case_lon'] = (df_grille['long'] // taille_grille) * taille_grille
    comptage_cases = df_grille.groupby(['case_lat', 'case_lon']).size()
    comptage_par_case = comptage_cases.to_dict()  # accès rapide : {(case_lat, case_lon): nb_accidents}

    # Cases occupées (au moins un accident)
    cases_occupees = set(comptage_par_case.keys())

    # Si demandé, on ajoute les cases voisines à zéro accident (dilatation), pour voir le contour
    # de la grille autour des zones concernées, sans couvrir tout le rectangle englobant
    cases_a_dessiner = set(cases_occupees)
    if rayon_voisinage > 0:
        for (case_lat, case_lon) in cases_occupees:
            for d_lat in range(-rayon_voisinage, rayon_voisinage + 1):
                for d_lon in range(-rayon_voisinage, rayon_voisinage + 1):
                    voisine = (round(case_lat + d_lat * taille_grille, 6), round(case_lon + d_lon * taille_grille, 6))
                    cases_a_dessiner.add(voisine)

    nb_cases = len(cases_a_dessiner)
    print(f"Cases avec accident(s): {len(cases_occupees)} | Cases dessinées: {nb_cases}")
    if nb_cases > 15000:
        print("ATTENTION : beaucoup de cases, la carte risque d'être lente à charger. "
              "Augmente 'taille_grille' ou réduis 'rayon_voisinage' pour réduire ce nombre.")

    # Échelle de couleur calée sur le nombre max d'accidents dans une case
    max_accidents = int(comptage_cases.max())
    echelle_couleur = cm.LinearColormap(
        colors=['yellow', 'orange', 'red', 'darkred'],
        vmin=1,
        vmax=max_accidents,
        caption="Nombre d'accidents par case"
    )

    for (lat, lon) in cases_a_dessiner:
        nb = comptage_par_case.get((lat, lon), 0)

        if nb == 0:
            # Case sans accident, affichée seulement pour le contour (rayon_voisinage > 0)
            couleur = '#cccccc'
            opacite = 0.08
        else:
            couleur = echelle_couleur(nb)
            opacite = 0.6

        folium.Rectangle(
            bounds=[[lat, lon], [lat + taille_grille, lon + taille_grille]],
            color=couleur,
            weight=1,
            fill=True,
            fillColor=couleur,
            fillOpacity=opacite,
            popup=f"{int(nb)} accident(s) dans cette zone"
        ).add_to(m5)

    echelle_couleur.add_to(m5)

    title_html5 = '''
                <h3 align="center" style="font-size:16px"><b>Densité d'accidents par quadrillage</b></h3>
                '''
    m5.get_root().html.add_child(folium.Element(title_html5))

    m5.save("maps/maritime_accidents_maps/accident_grid.html")
    
def accident_grille_temps(taille_grille=2):
    """carte avec un quadrillage avec un curseur de temps"""
    df_grille = df.copy()
    df_grille['case_lat'] = (df_grille['lat'] // taille_grille) * taille_grille
    df_grille['case_lon'] = (df_grille['long'] // taille_grille) * taille_grille

    # Compte le nombre d'accidents par (année, case)
    comptage = df_grille.groupby(['annee', 'case_lat', 'case_lon']).size().reset_index(name='nb_accidents')

    max_accidents = int(comptage['nb_accidents'].max())
    echelle_couleur = cm.LinearColormap(
        colors=['yellow', 'orange', 'red', 'darkred'],
        vmin=1,
        vmax=max_accidents,
        caption="Nombre d'accidents par case et par année"
    )

    features = []
    for _, row in comptage.iterrows():
        lat, lon, nb, annee = row['case_lat'], row['case_lon'], row['nb_accidents'], int(row['annee'])
        couleur = echelle_couleur(nb)

        polygone = [[
            [lon, lat],
            [lon + taille_grille, lat],
            [lon + taille_grille, lat + taille_grille],
            [lon, lat + taille_grille],
            [lon, lat],
        ]]

        feature = {
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': polygone},
            'properties': {
                'time': f"{annee}-01-01",
                'style': {
                    'color': couleur,
                    'fillColor': couleur,
                    'fillOpacity': 0.6,
                    'weight': 1,
                },
                'popup': f"{int(nb)} accident(s) en {annee}",
            }
        }
        features.append(feature)

    geojson_data = {'type': 'FeatureCollection', 'features': features}

    m6 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    TimestampedGeoJson(
        geojson_data,
        period='P1Y',
        duration='P1D',
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=2,
        loop_button=True,
        date_options='YYYY',
        time_slider_drag_update=True
    ).add_to(m6)

    echelle_couleur.add_to(m6)

    title_html6 = '''
                <h3 align="center" style="font-size:16px"><b>Évolution du quadrillage des accidents par année</b></h3>
                <p align="center" style="font-size:12px">Utilise le curseur en bas pour voir l'évolution année par année</p>
                '''
    m6.get_root().html.add_child(folium.Element(title_html6))

    m6.save("maps/maritime_accidents_maps/accident_grid_timeline.html")
 
def accident_filtrable_intersection():
    """"""

    df_carte = df.dropna(subset=['gravite', 'bateau']).copy()
    points_json = df_carte[['lat', 'long', 'gravite', 'bateau']].to_dict(orient='records')

    gravites = sorted(df_carte['gravite'].unique())
    bateaux = sorted(df_carte['bateau'].unique())

    m8 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    nom_carte = m8.get_name()  # nom de la variable JS générée par Folium pour cette carte

    points_js = json.dumps(points_json)
    couleurs_js = json.dumps(couleurs)
    
    cases_gravite = "".join(
        f'<label style="display:block"><input type="checkbox" class="filtre-gravite" value="{g}" checked> {g}</label>'
        for g in gravites
    )
    cases_bateau = "".join(
        f'<label style="display:block"><input type="checkbox" class="filtre-bateau" value="{b}" checked> {b}</label>'
        for b in bateaux
    )

    html_controle = f'''
    <div style="position: fixed; top: 80px; right: 10px; z-index: 1000; background: white;
                padding: 10px; border: 2px solid grey; border-radius: 5px; max-height: 70vh;
                overflow-y: auto; font-size: 13px;">
        <b>Gravité</b><br>{cases_gravite}
        <hr>
        <b>Bateau</b><br>{cases_bateau}
        <hr>
        <span id="compteur-filtre">{len(points_json)} accident(s) affiché(s)</span>
    </div>
    '''

    # window.addEventListener('load', ...) : Folium génère lui-même un script qui crée la carte
    # (la variable {nom_carte}), et ce script peut s'exécuter APRÈS le nôtre selon l'ordre de rendu
    # de la page. Sans ce garde-fou, notre script plantait silencieusement dès la référence à la
    # carte (qui n'existait pas encore), donc rien ne se dessinait ni ne réagissait aux cases à
    # cocher. En attendant l'événement "load" (déclenché une fois TOUTE la page chargée), on est
    # sûr que la carte existe déjà.
    script = f'''
    <script>
    window.addEventListener('load', function() {{
        var pointsData = {points_js};
        var couleursGravite = {couleurs_js};
        var coucheAccidents9 = L.layerGroup().addTo({nom_carte});
        var marqueurs9 = [];

        pointsData.forEach(function(p) {{
            var couleur = couleursGravite[p.gravite] || 'gray';
            var marqueur = L.circleMarker([p.lat, p.long], {{
                radius: 4, color: couleur, fillColor: couleur, fillOpacity: 0.7, weight: 1
            }});
            marqueur.gravite = p.gravite;
            marqueur.bateau = p.bateau;
            marqueur.bindTooltip(p.gravite + " - " + p.bateau);
            marqueurs9.push(marqueur);
            marqueur.addTo(coucheAccidents9);
        }});

        function appliquerFiltre9() {{
            var gravitesCochees = Array.from(document.querySelectorAll('.filtre-gravite:checked')).map(cb => cb.value);
            var bateauxCoches = Array.from(document.querySelectorAll('.filtre-bateau:checked')).map(cb => cb.value);
            var nbAffiches = 0;

            marqueurs9.forEach(function(m) {{
                var correspond = gravitesCochees.includes(m.gravite) && bateauxCoches.includes(m.bateau);
                if (correspond) {{
                    if (!coucheAccidents9.hasLayer(m)) coucheAccidents9.addLayer(m);
                    nbAffiches++;
                }} else {{
                    if (coucheAccidents9.hasLayer(m)) coucheAccidents9.removeLayer(m);
                }}
            }});
            document.getElementById('compteur-filtre').innerText = nbAffiches + " accident(s) affiché(s)";
        }}

        document.querySelectorAll('.filtre-gravite, .filtre-bateau').forEach(function(cb) {{
            cb.addEventListener('change', appliquerFiltre9);
        }});

        // Applique le filtre une première fois au chargement, pour que le compteur soit juste
        // dès le départ (avant même le premier clic sur une case)
        appliquerFiltre9();
    }});
    </script>
    '''

    m8.get_root().html.add_child(folium.Element(html_controle))
    m8.get_root().html.add_child(folium.Element(script))

    title_html8 = '''
                <h3 align="center" style="font-size:16px"><b>Accidents filtrables (intersection gravité ET bateau)</b></h3>
                <p align="center" style="font-size:12px">Coche/décoche à droite : seuls les accidents correspondant aux DEUX filtres s'affichent</p>
                '''
    m8.get_root().html.add_child(folium.Element(title_html8))

    m8.save("maps/maritime_accidents_maps/accident_filter_intersection.html")

    
    
accidents()
heatmap()
accident_annee()
accident_temps_animation()
accident_grille()
accident_grille_temps()
accident_filtrable_intersection()