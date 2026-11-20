"""Telecharge le jeu de donnees Kaggle et le copie dans data/raw.csv."""
import glob
import os
import shutil

import kagglehub
import pandas as pd

from app.config import DATA_DIR, RAW_CSV

path = kagglehub.dataset_download("saifulislam7/cyberbullying-and-harassment-detection-using-ml")
print("Dossier du dataset :", path)

csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csvs:
    raise SystemExit("Aucun fichier CSV trouve dans le dataset.")
csvs.sort(key=os.path.getsize, reverse=True)
print("Fichiers CSV :", csvs)

DATA_DIR.mkdir(exist_ok=True)
shutil.copy(csvs[0], RAW_CSV)
print("Copie :", csvs[0], "->", RAW_CSV)

df = pd.read_csv(RAW_CSV)
print("Colonnes :", list(df.columns))
print(df.head())
