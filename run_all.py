"""Lance le pipeline complet : nettoyage -> MongoDB -> analyse -> modeles.
Usage : python run_all.py [--skip-mongo]"""
import subprocess
import sys

skip_mongo = "--skip-mongo" in sys.argv
steps = [("1. Nettoyage", "app.cleaning"), ("2. Chargement MongoDB", "app.db"),
         ("3. Analyse descriptive + figures", "app.analysis"), ("4-6. Modeles, comparaison, figures", "app.modeling")]

for title, module in steps:
    if skip_mongo and module == "app.db":
        continue
    print(f"\n########## {title} ##########")
    code = subprocess.call([sys.executable, "-m", module])
    if code != 0:
        if module == "app.db":
            print("!! MongoDB indisponible : verifiez 'docker ps'. On continue avec data/clean.csv.")
            continue
        raise SystemExit(f"Echec a l'etape : {title}")
print("\nPipeline termine.")
