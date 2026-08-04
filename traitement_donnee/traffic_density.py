import pandas as pd
import glob
import os

folder = "vessel_density_data"
output_file = os.path.join(folder, "global_density.csv")

dfs = []
for filepath in glob.glob(os.path.join(folder, "*.csv")):
    if os.path.basename(filepath) == "global_density.csv":
        continue

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  

    if 'vd' not in df.columns:
        print(f"Colonne 'vd' absente dans {os.path.basename(filepath)}, fichier ignoré.")
        continue

    if 'latitude' not in df.columns and 'y' in df.columns:
        df = df.rename(columns={'y': 'latitude'})
    if 'longitude' not in df.columns and 'x' in df.columns:
        df = df.rename(columns={'x': 'longitude'})

    df = df.drop(columns=[c for c in ['x', 'y'] if c in df.columns])

    df['vd'] = pd.to_numeric(df['vd'], errors='coerce')

    before = len(df)
    df = df.dropna(subset=['vd'])
    df = df[df['vd'] != 0.0]
    after = len(df)

    cols_order = [c for c in ['time', 'vd', 'latitude', 'longitude'] if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + other_cols]
    dfs.append(df)

if dfs:
    result = pd.concat(dfs, ignore_index=True)
    result.to_csv("data/global_density.csv", index=False)
else:
    print("Aucun fichier CSV avec une colonne 'vd' trouvé.")