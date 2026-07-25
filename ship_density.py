import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import numpy as np

CHEMIN_TIF = "shipdensity_global.tif"

with rasterio.open(CHEMIN_TIF) as src:
    # --- 1. Métadonnées : instantané, ne lit aucun pixel ---
    print("=== Métadonnées ===")
    print("Largeur x Hauteur :", src.width, "x", src.height)
    print("Nombre de bandes  :", src.count)
    print("CRS (projection)  :", src.crs)
    print("Résolution        :", src.res)
    print("Emprise (bounds)  :", src.bounds)
    print("Type de données   :", src.dtypes)
    print("Valeur nodata     :", src.nodata)
    print("Overviews dispo   :", src.overviews(1))  # niveaux de zoom pré-calculés, si présents

    # --- 2. Aperçu basse résolution ---
    # out_shape sous-échantillonne PENDANT la lecture (rasterio ne charge que ce qu'il faut,
    # pas les 9 Go en mémoire). On vise ~1000 px de large pour un aperçu rapide.
    facteur = max(1, src.width // 1000)
    largeur_preview = src.width // facteur
    hauteur_preview = src.height // facteur

    print(f"\nLecture d'un aperçu {largeur_preview}x{hauteur_preview} (facteur /{facteur})...")
    apercu = src.read(
        1,  # bande 1
        out_shape=(hauteur_preview, largeur_preview),
        resampling=Resampling.average  # moyenne les pixels regroupés (mieux que nearest pour de la densité)
    )

    # Masque les valeurs nodata pour ne pas les afficher comme si c'était du trafic réel
    if src.nodata is not None:
        apercu_masque = np.ma.masked_equal(apercu, src.nodata)
    else:
        apercu_masque = apercu

    print("Valeurs min/max sur l'aperçu :", apercu_masque.min(), apercu_masque.max())

    # --- 3. Affichage ---
    plt.figure(figsize=(14, 7))
    plt.imshow(apercu_masque, cmap='inferno')
    plt.colorbar(label="Densité de trafic")
    plt.title("Aperçu basse résolution : densité de trafic maritime mondial")
    plt.tight_layout()
    plt.savefig("apercu_ship_density.png", dpi=150)
    print("\nAperçu sauvegardé dans apercu_ship_density.png")
    plt.show()