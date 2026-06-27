import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
dossier = 'données_d_accident_en_mer/'
fichiers = glob.glob(os.path.join(dossier, '*.csv'))
# df_trie = df.sort_values('Date of occurrence')
#print(df.info())         # Infos sur les colonnes
#print(df.describe())     # Statistiques descriptives
# print(df_trie)

