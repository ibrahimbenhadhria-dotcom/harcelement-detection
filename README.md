# Detection de harcelement social : test Data Science

Pipeline complet : nettoyage, MongoDB, analyse descriptive, 2 modeles, etude comparative, visualisations, API FastAPI + Docker.
Jeu de donnees : Kaggle `saifulislam7/cyberbullying-and-harassment-detection-using-ml`.

## Structure

```
app/
  config.py          chemins et parametres (MONGO_URI, etc.)
  text_utils.py      nettoyage de texte (partage entrainement / API)
  cleaning.py        etape 1 : exploration + nettoyage -> data/clean.csv
  db.py              etape 2 : chargement dans MongoDB
  analysis.py        etape 3 : analyse descriptive
  modeling.py        etapes 4-5 : 2 modeles + etude comparative
  visualization.py   etape 6 : figures
  main.py            etape 7 : API FastAPI
download_data.py     telechargement Kaggle -> data/raw.csv
run_all.py           lance les etapes 1 a 6
Dockerfile, docker-compose.yml, requirements*.txt
reports/             analyse_descriptive.md, etude_comparative.md, figures/
models/              modeles entraines + metrics.json
```

## Prerequis

- Python 3.11+
- Docker Desktop (pour MongoDB et l'etape 7)
- Un compte Kaggle gratuit (le telechargement peut demander une connexion)

## Execution (Windows PowerShell)

```powershell
cd C:\projets\harcelement
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# MongoDB (une seule fois, ensuite : docker start mongo-harcelement)
docker run -d --name mongo-harcelement -p 27017:27017 -v mongo_data:/data/db mongo:7

python download_data.py          # telecharge -> data/raw.csv
python -m app.cleaning           # etape 1
python -m app.db                 # etape 2
python -m app.analysis           # etape 3 (+ figures)
python -m app.modeling           # etapes 4-5-6 (+ etude comparative)
```

Ou tout d'un coup : `python run_all.py` (ajoutez `--skip-mongo` si MongoDB n'est pas lance).

Verifier MongoDB :

```powershell
docker exec -it mongo-harcelement mongosh
use harcelement
db.messages.countDocuments({})
db.messages.findOne()
```

Si MongoDB est injoignable, les etapes 3 et 4 lisent `data/clean.csv` a la place.

## Resultats produits

- `reports/analyse_descriptive.md` et figures `01` a `04`
- `reports/etude_comparative.md` (livrable 4) et figures `05` a `08`
- `models/best_model.joblib` : meilleur modele selon le F1

## API FastAPI (local)

```powershell
uvicorn app.main:app --reload
```

Documentation interactive : http://localhost:8000/docs

```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"you are a pathetic loser\"}"
```

Reponse : `{"label": 1, "label_name": "harcelement", "probability_harassment": 0.97, "model": "..."}`

Autres routes : `GET /health`, `GET /metrics`.

## Deploiement avec Docker

L'entrainement doit etre fait avant (le dossier `models/` est copie dans l'image).

```powershell
docker build -t harcelement-api .
docker run -d --name harcelement-api -p 8000:8000 harcelement-api
```

Ou avec docker-compose (API + MongoDB sur le port 27018) :

```powershell
docker compose up -d --build
```

Test : http://localhost:8000/health

Pour changer MongoDB : variable d'environnement `MONGO_URI` (ex. `mongodb://mongo:27017`).

## Choix techniques

- **Nettoyage** : minuscules, suppression URL/mentions/emojis/chiffres, doublons, textes ambigus (meme texte, labels differents), normalisation des labels en 0/1. Prevu pour des textes en anglais.
- **Modeles** : TF-IDF + Regression logistique, et TF-IDF + Naive Bayes multinomial. Hyperparametres par GridSearchCV (F1, 3 plis), decoupage 80/20 stratifie.
- **Metriques** : accuracy, precision, recall, F1, ROC-AUC, matrice de confusion, temps.
