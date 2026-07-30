import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from folium.plugins import TimestampedGeoJson
from pyproj import Transformer
import json
import os

df = pd.read_csv(os.path.join("data", "global_density.csv"))
df.columns = df.columns.str.strip()
df['vd'] = pd.to_numeric(df['vd'], errors='coerce')
df = df.dropna(subset=['vd', 'latitude', 'longitude', 'time'])

CRS_SOURCE = "EPSG:3035"
TAILLE_CASE_M = 20000  # pas de la grille source, en mètres (déduit des données : 20 km)

transformer = Transformer.from_crs(CRS_SOURCE, "EPSG:4326", always_xy=True)


def densite_flotte(facteur_agregation=6):
    """Carte : densité du trafic maritime, quadrillage réel (grille source reprojetée), curseur de temps

    facteur_agregation : regroupe les cases natives (20 km) par blocs de N x N pour réduire le
    nombre de polygones (446 760 cases natives -> fichier HTML de ~200 Mo, illisible pour un
    navigateur). Avec facteur_agregation=3, les cases font 60 km de côté et le nombre de
    polygones est divisé par ~9.
    """
    taille_case = TAILLE_CASE_M * facteur_agregation
    demi = taille_case / 2

    # Regroupe les cases natives sur une grille plus grossière, moyenne de vd par (time, case)
    df_agg = df.copy()
    df_agg['case_x'] = (np.round(df_agg['longitude'] / taille_case) * taille_case)
    df_agg['case_y'] = (np.round(df_agg['latitude'] / taille_case) * taille_case)
    df_agg = (
        df_agg.groupby(['time', 'case_x', 'case_y'])['vd']
        .mean()
        .reset_index()
    )
    print(f"Cases natives : {len(df)} -> cases agrégées : {len(df_agg)}")

    # Coins de chaque case en coordonnées projetées (m), calculés d'un coup pour toutes les lignes
    x = df_agg['case_x'].values
    y = df_agg['case_y'].values

    coins_x = np.stack([x - demi, x + demi, x + demi, x - demi], axis=1)
    coins_y = np.stack([y - demi, y - demi, y + demi, y + demi], axis=1)

    # Reprojection vectorisée de tous les coins en une seule fois (rapide)
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

    # Arrondi des coordonnées (4 décimales ~ 11 m, largement suffisant) pour réduire la taille du HTML
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

    m1.save("carte_flotte/carte_densite_flotte.html")
    print("Carte enregistrée : carte_densite_flotte.html")


densite_flotte()