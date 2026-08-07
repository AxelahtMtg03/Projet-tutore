import pandas as pd
import glob
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "vessel_density"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def traiter_densite_trafic():
    output_file = PROCESSED_DIR / "vessel_density.csv"

    dfs = []
    for filepath in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        if os.path.basename(filepath) == "global_density.csv":
            continue

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        if 'vd' not in df.columns:
            continue

        if 'latitude' not in df.columns and 'y' in df.columns:
            df = df.rename(columns={'y': 'latitude'})
        if 'longitude' not in df.columns and 'x' in df.columns:
            df = df.rename(columns={'x': 'longitude'})

        df = df.drop(columns=[c for c in ['x', 'y'] if c in df.columns])

        df['vd'] = pd.to_numeric(df['vd'], errors='coerce')

        df = df.dropna(subset=['vd'])
        df = df[df['vd'] != 0.0]

        cols_order = [c for c in ['time', 'vd', 'latitude', 'longitude'] if c in df.columns]
        other_cols = [c for c in df.columns if c not in cols_order]
        df = df[cols_order + other_cols]
        dfs.append(df)

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        result.to_csv(output_file, index=False)
        return result
    else:
        print("Aucun fichier CSV avec une colonne 'vd' trouvé.")
        return None

traiter_densite_trafic()