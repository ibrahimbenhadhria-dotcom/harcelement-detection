"""Etape 7 : API FastAPI de prediction."""
import json
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import MODELS_DIR
from app.text_utils import clean_text

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = MODELS_DIR / "best_model.joblib"
    meta_path = MODELS_DIR / "metrics.json"
    if model_path.exists():
        state["model"] = joblib.load(model_path)
    if meta_path.exists():
        state["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    yield
    state.clear()


app = FastAPI(title="API de detection de harcelement social", version="1.0.0", lifespan=lifespan)


class Message(BaseModel):
    text: str = Field(..., min_length=1, examples=["You are so stupid, nobody likes you"])


class Prediction(BaseModel):
    label: int
    label_name: str
    probability_harassment: float
    model: str


def _model():
    if "model" not in state:
        raise HTTPException(503, "Modele absent. Lancez l'entrainement (python -m app.modeling).")
    return state["model"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/metrics")
def metrics():
    if "meta" not in state:
        raise HTTPException(503, "metrics.json absent.")
    return state["meta"]


@app.post("/predict", response_model=Prediction)
def predict(msg: Message):
    model = _model()
    cleaned = clean_text(msg.text)
    if not cleaned:
        raise HTTPException(422, "Le message ne contient aucun mot exploitable apres nettoyage.")
    proba = float(model.predict_proba([cleaned])[0][1])
    label = int(proba >= 0.5)
    return Prediction(label=label, label_name="harcelement" if label else "non_harcelement",
                      probability_harassment=round(proba, 4),
                      model=state.get("meta", {}).get("best_model", "unknown"))
