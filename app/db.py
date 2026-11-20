"""Etape 2 : chargement des donnees nettoyees dans MongoDB (et relecture)."""
import pandas as pd
from pymongo import ASCENDING, MongoClient

from app.config import CLEAN_CSV, MONGO_COLLECTION, MONGO_DB, MONGO_URI


def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
    client.admin.command("ping")  # leve une exception si MongoDB est injoignable
    return client[MONGO_DB][MONGO_COLLECTION]


def load_clean_to_mongo(drop: bool = True, batch_size: int = 10000) -> int:
    if not CLEAN_CSV.exists():
        raise SystemExit(f"{CLEAN_CSV} introuvable. Lancez d'abord : python -m app.cleaning")
    df = pd.read_csv(CLEAN_CSV)
    col = get_collection()
    if drop:
        col.drop()
    records = df.to_dict("records")
    for i in range(0, len(records), batch_size):
        col.insert_many(records[i : i + batch_size])
    col.create_index([("label", ASCENDING)])
    col.create_index([("type", ASCENDING)])
    return col.count_documents({})


def read_from_mongo() -> pd.DataFrame:
    col = get_collection()
    df = pd.DataFrame(list(col.find({}, {"_id": 0})))
    if df.empty:
        raise ValueError("Collection MongoDB vide.")
    return df


def load_dataset(prefer_mongo: bool = True) -> pd.DataFrame:
    """Lit les donnees depuis MongoDB, sinon depuis data/clean.csv."""
    if prefer_mongo:
        try:
            df = read_from_mongo()
            print(f"Source des donnees : MongoDB ({MONGO_DB}.{MONGO_COLLECTION}), {len(df)} documents")
            return df
        except Exception as exc:  # noqa: BLE001
            print(f"MongoDB indisponible ({type(exc).__name__}) -> lecture de {CLEAN_CSV.name}")
    df = pd.read_csv(CLEAN_CSV)
    print(f"Source des donnees : {CLEAN_CSV.name}, {len(df)} lignes")
    return df


def main() -> None:
    n = load_clean_to_mongo()
    print(f"{n} documents inseres dans {MONGO_DB}.{MONGO_COLLECTION}")
    print("Exemple :", get_collection().find_one({}, {"_id": 0}))


if __name__ == "__main__":
    main()
