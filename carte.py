import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson

df = pd.read_csv("accidents_nettoyes.csv")


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

# CARTE 1 : Tous les accidents 
def accidents():
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
        # Choisir la couleur selon la gravité
        color = couleurs.get(row['gravite'], 'gray')         
        # Créer le popup avec les infos
        popup_text = f"""
        <b>Date:</b> {row['Date of occurrence']}<br>
        <b>Heure:</b> {row['heure']}h<br>
        <b>Gravité:</b> {row['gravite']}<br>
        <b>Port:</b> {row['port']}<br>
        <b>Bateau:</b> {row['bateau']}<br>
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

    # Ajouter une légende
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><b>Légende - Gravité</b></p>
        <p><span style="color: red;">●</span> Very Serious</p>
        <p><span style="color: orange;">●</span> Sérieux</p>
        <p><span style="color: blue;">●</span> Marine incident</p>
        <p><span style="color: gray;">●</span> less serious</p>
    </div>
    '''
    m1.get_root().html.add_child(folium.Element(legend_html))

    m1.save("carte_folium/carte_accidents.html")
    
def heatmap():
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
    # CARTE 3 : Accidents par année (avec couches)
    m3 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    # Créer un groupe pour chaque année
    annees = sorted(df['annee'].unique())
    feature_groups = {}

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

    # Ajouter le contrôle des couches
    folium.LayerControl().add_to(m3)

    title_html3 = '''
                <h3 align="center" style="font-size:16px"><b>Accidents par année</b></h3>
                '''
    m3.get_root().html.add_child(folium.Element(title_html3))

    m3.save("carte_accidents_par_annee.html")

# CARTE avec curseur de temps
def accident_temps_animation():
    df['date_str'] = pd.to_datetime(df['Date of occurrence']).dt.strftime('%Y-%m-%d %H:%M:%S')

    features = []
    for idx, row in df.iterrows():
        # Déterminer la couleur selon la gravité
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

    # Ajouter le plugin TimestampedGeoJson
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

    m4.save("carte_accidents_timeline.html")

accidents()