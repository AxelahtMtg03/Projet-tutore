import pandas as pd
import folium
import branca.colormap as cm
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson
from charge import charger_donnees_finales

df = charger_donnees_finales()


df = df.dropna(subset=['lat', 'long'])
print(f"Nombre d'accidents avec coordonnées: {len(df)}")

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
        # Crée le contenu du popup affiché au clic sur un point
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

    m1.save("carte_folium/carte_accidents.html")
    
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

    m2.save("carte_folium/carte_accidents_heatmap.html")

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

    m3.save("carte_folium/carte_accidents_par_annee.html")

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

    m4.save("carte_folium/carte_accidents_timeline.html")

def accident_grille(taille_grille=1):
    """Carte avec un quadrillage en fonction du nb d'accident dans une zone
    """
    df_clip = df[
        df['lat'].between(df['lat'].quantile(0.01), df['lat'].quantile(0.99)) &
        df['long'].between(df['long'].quantile(0.01), df['long'].quantile(0.99))
    ]

    m5 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    # Affecte chaque accident à une case de la grille (arrondi vers le bas au multiple de taille_grille)
    df_grille = df_clip.copy()
    df_grille['case_lat'] = (df_grille['lat'] // taille_grille) * taille_grille
    df_grille['case_lon'] = (df_grille['long'] // taille_grille) * taille_grille
    comptage_cases = df_grille.groupby(['case_lat', 'case_lon']).size()
    comptage_par_case = comptage_cases.to_dict()  # accès rapide : {(case_lat, case_lon): nb_accidents}

    # Bornes de la grille complète (arrondies aux multiples de taille_grille les plus proches)
    lat_min = (df_clip['lat'].min() // taille_grille) * taille_grille
    lat_max = (df_clip['lat'].max() // taille_grille + 1) * taille_grille
    lon_min = (df_clip['long'].min() // taille_grille) * taille_grille
    lon_max = (df_clip['long'].max() // taille_grille + 1) * taille_grille

    nb_lignes = round((lat_max - lat_min) / taille_grille)
    nb_colonnes = round((lon_max - lon_min) / taille_grille)
    nb_cases = nb_lignes * nb_colonnes
    print(f"Grille : {nb_lignes} x {nb_colonnes} = {nb_cases} cases à dessiner")
    if nb_cases > 15000:
        print("ATTENTION : beaucoup de cases, la carte risque d'être lente à charger. "
              "Augmente 'taille_grille' pour réduire ce nombre.")

    # Échelle de couleur calée sur le nombre max d'accidents dans UNE case (les cases à 0 sont
    # traitées à part plus bas, elles ne suivent pas cette échelle)
    max_accidents = int(comptage_cases.max())
    echelle_couleur = cm.LinearColormap(
        colors=['yellow', 'orange', 'red', 'darkred'],
        vmin=1,
        vmax=max_accidents,
        caption="Nombre d'accidents par case"
    )

    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            nb = comptage_par_case.get((round(lat, 6), round(lon, 6)), 0)

            if nb == 0:
                # Case sans accident : grise et très transparente, juste pour visualiser le quadrillage
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

            lon += taille_grille
        lat += taille_grille

    echelle_couleur.add_to(m5)

    title_html5 = '''
                <h3 align="center" style="font-size:16px"><b>Densité d'accidents par quadrillage</b></h3>
                '''
    m5.get_root().html.add_child(folium.Element(title_html5))

    m5.save("carte_folium/carte_accidents_grille.html")

accidents()
# heatmap()
# accident_annee()
# accident_temps_animation()
# accident_grille()