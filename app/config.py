"""Configuration centrale du projet."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw.csv"
CLEAN_CSV = DATA_DIR / "clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "harcelement")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "messages")

RANDOM_STATE = 42
TEST_SIZE = 0.2
