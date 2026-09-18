import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from folium.plugins import TimestampedGeoJson
from pyproj import Transformer
import os

df = pd.read_csv("data/processed/vessel_density.csv")
df.columns = df.columns.str.strip()
df['vd'] = pd.to_numeric(df['vd'], errors='coerce')
df = df.dropna(subset=['vd', 'latitude', 'longitude', 'time'])

CRS_SOURCE = "EPSG:3035"
TAILLE_CASE_M = 20000 
transformer = Transformer.from_crs(CRS_SOURCE, "EPSG:4326", always_xy=True)

def densite_flotte(facteur_agregation=6):

    taille_case = TAILLE_CASE_M * facteur_agregation
    demi = taille_case / 2

    df_agg = df.copy()
    df_agg['case_x'] = (np.round(df_agg['longitude'] / taille_case) * taille_case)
    df_agg['case_y'] = (np.round(df_agg['latitude'] / taille_case) * taille_case)
    df_agg = (
        df_agg.groupby(['time', 'case_x', 'case_y'])['vd']
        .mean()
        .reset_index()
    )

    x = df_agg['case_x'].values
    y = df_agg['case_y'].values

    coins_x = np.stack([x - demi, x + demi, x + demi, x - demi], axis=1)
    coins_y = np.stack([y - demi, y - demi, y + demi, y + demi], axis=1)

    coins_lon, coins_lat = transformer.transform(coins_x.ravel(), coins_y.ravel())
    coins_lon = coins_lon.reshape(coins_x.shape)
    coins_lat = coins_lat.reshape(coins_y.shape)

    vmax = df_agg['vd'].quantile(0.99)
    echelle_couleur = cm.LinearColormap(
        colors=['yellow', 'orange', 'red', 'darkred'],
        vmin=0,
        vmax=vmax,
        caption="Densité (vd)"
    )

    center_lon_c, center_lat_c = transformer.transform(x.mean(), y.mean())

    vd = df_agg['vd'].values
    times = df_agg['time'].astype(str).values

    coins_lon = np.round(coins_lon, 4)
    coins_lat = np.round(coins_lat, 4)

    features = []
    for i in range(len(df_agg)):
        v = min(vd[i], vmax)
        couleur = echelle_couleur(v)
        polygone = [[
            [coins_lon[i, 0], coins_lat[i, 0]],
            [coins_lon[i, 1], coins_lat[i, 1]],
            [coins_lon[i, 2], coins_lat[i, 2]],
            [coins_lon[i, 3], coins_lat[i, 3]],
            [coins_lon[i, 0], coins_lat[i, 0]],
        ]]
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': polygone},
            'properties': {
                'time': times[i],
                'style': {
                    'color': couleur,
                    'fillColor': couleur,
                    'fillOpacity': 0.7,
                    'weight': 0,
                },
                'popup': f"vd: {v:.3f}",
            }
        })

    geojson_data = {'type': 'FeatureCollection', 'features': features}

    m1 = folium.Map(
        location=[center_lat_c, center_lon_c],
        zoom_start=4,
        tiles='OpenStreetMap'
    )

    TimestampedGeoJson(
        geojson_data,
        period='P1M',  # les 'time' du CSV sont mensuels (ex: 2017-01-01, 2017-02-01, ...)
        duration='P1M',
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=2,
        loop_button=True,
        date_options='YYYY-MM',
        time_slider_drag_update=True
    ).add_to(m1)

    echelle_couleur.add_to(m1)

    title_html = '''
                <h3 align="center" style="font-size:16px"><b>Densité du trafic maritime</b></h3>
                <p align="center" style="font-size:12px">Utilise le curseur en bas pour voir l'évolution mois par mois</p>
                '''
    m1.get_root().html.add_child(folium.Element(title_html))

    m1.save("maps/vessel_density_maps/vessel_density_curved_grid.html")

def densite_flotte2(taille_grille_deg=3):
    lon_deg, lat_deg = transformer.transform(df['longitude'].values, df['latitude'].values)
    df_deg = df.copy()
    df_deg['lon_deg'] = lon_deg
    df_deg['lat_deg'] = lat_deg

    df_deg['case_lat'] = (df_deg['lat_deg'] // taille_grille_deg) * taille_grille_deg
    df_deg['case_lon'] = (df_deg['lon_deg'] // taille_grille_deg) * taille_grille_deg

    df_agg = (
        df_deg.groupby(['time', 'case_lat', 'case_lon'])['vd']
        .mean()
        .reset_index()
    )
    print(f"Cases natives : {len(df)} -> cases agrégées : {len(df_agg)}")

    vmax = df_agg['vd'].quantile(0.99)
    echelle_couleur = cm.LinearColormap(
        colors=['yellow', 'orange', 'red', 'darkred'],
        vmin=0,
        vmax=vmax,
        caption="Densité (vd)"
    )

    center_lat_c = df_agg['case_lat'].mean() + taille_grille_deg / 2
    center_lon_c = df_agg['case_lon'].mean() + taille_grille_deg / 2

    features = []
    for row in df_agg.itertuples(index=False):
        v = min(row.vd, vmax)
        couleur = echelle_couleur(v)
        lat, lon = row.case_lat, row.case_lon

        # Rectangle toujours droit : simple carré en degrés, comme accident_grille() dans carte.py
        polygone = [[
            [lon, lat],
            [lon + taille_grille_deg, lat],
            [lon + taille_grille_deg, lat + taille_grille_deg],
            [lon, lat + taille_grille_deg],
            [lon, lat],
        ]]
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': polygone},
            'properties': {
                'time': str(row.time),
                'style': {
                    'color': couleur,
                    'fillColor': couleur,
                    'fillOpacity': 0.7,
                    'weight': 0,
                },
                'popup': f"vd: {v:.3f}",
            }
        })

    geojson_data = {'type': 'FeatureCollection', 'features': features}

    m2 = folium.Map(
        location=[center_lat_c, center_lon_c],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    TimestampedGeoJson(
        geojson_data,
        period='P1M',  # les 'time' du CSV sont mensuels (ex: 2017-01-01, 2017-02-01, ...)
        duration='P1M',
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=2,
        loop_button=True,
        date_options='YYYY-MM',
        time_slider_drag_update=True
    ).add_to(m2)

    echelle_couleur.add_to(m2)

    title_html = '''
                <h3 align="center" style="font-size:16px"><b>Densité du trafic maritime</b></h3>
                <p align="center" style="font-size:12px">Utilise le curseur en bas pour voir l'évolution mois par mois</p>
                '''
    m2.get_root().html.add_child(folium.Element(title_html))

    m2.save("maps/vessel_density_maps/vessel_density_grid.html")

def densite_flotte3(taille_grille_deg=3):
    """Carte statique : densité moyenne sur toute la période (pas de curseur de temps)"""
    lon_deg, lat_deg = transformer.transform(df['longitude'].values, df['latitude'].values)
    df_deg = df.copy()
    df_deg['lon_deg'] = lon_deg
    df_deg['lat_deg'] = lat_deg

    df_deg['case_lat'] = (df_deg['lat_deg'] // taille_grille_deg) * taille_grille_deg
    df_deg['case_lon'] = (df_deg['lon_deg'] // taille_grille_deg) * taille_grille_deg

    # Pas de groupby sur 'time' ici : on moyenne directement toute la période
    df_agg = (
        df_deg.groupby(['case_lat', 'case_lon'])['vd']
        .mean()
        .reset_index()
    )
    print(f"Cases natives : {len(df)} -> cases agrégées : {len(df_agg)}")

    vmax = df_agg['vd'].quantile(0.99)
    echelle_couleur = cm.LinearColormap(
        colors=['yellow', 'orange', 'red', 'darkred'],
        vmin=0,
        vmax=vmax,
        caption="Densité moyenne (vd)"
    )

    center_lat_c = df_agg['case_lat'].mean() + taille_grille_deg / 2
    center_lon_c = df_agg['case_lon'].mean() + taille_grille_deg / 2

    m3 = folium.Map(
        location=[center_lat_c, center_lon_c],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    for row in df_agg.itertuples(index=False):
        v = min(row.vd, vmax)
        couleur = echelle_couleur(v)
        lat, lon = row.case_lat, row.case_lon

        folium.Rectangle(
            bounds=[[lat, lon], [lat + taille_grille_deg, lon + taille_grille_deg]],
            color=couleur,
            weight=0,
            fill=True,
            fillColor=couleur,
            fillOpacity=0.7,
            popup=f"vd moyen: {v:.3f}"
        ).add_to(m3)

    echelle_couleur.add_to(m3)

    title_html = '''
                <h3 align="center" style="font-size:16px"><b>Densité globale du trafic maritime</b></h3>
                <p align="center" style="font-size:12px">Moyenne sur toute la période disponible (pas de curseur temporel)</p>
                '''
    m3.get_root().html.add_child(folium.Element(title_html))

    m3.save("maps/vessel_density_maps/vessel_density_global.html")
# densite_flotte()
# densite_flotte2(taille_grille_deg=0.5)
densite_flotte3(taille_grille_deg=0.5)