"""Etape 1 : exploration et nettoyage du jeu de donnees."""
import re

import numpy as np
import pandas as pd

from app.config import CLEAN_CSV, RAW_CSV
from app.text_utils import clean_text

TEXT_CANDIDATES = ["text", "tweet_text", "tweet", "message", "comment", "content"]
LABEL_CANDIDATES = ["label", "labels", "class", "target"]
TYPE_CANDIDATES = ["types", "type", "category", "cyberbullying_type"]


def find_col(df: pd.DataFrame, candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def read_raw(path=RAW_CSV) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Impossible de lire le CSV (encodage inconnu).")


def normalize_label(value):
    """Convertit les variantes de labels en 1 (harcelement) / 0 (non)."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if s in {"0", "0.0", "false", "no", "n"}:
        return 0
    if s.startswith("non") or re.search(r"\b(non|not|no|neutral|normal|safe|clean)\b", s):
        return 0
    return 1


def explore(df: pd.DataFrame) -> None:
    print("=== EXPLORATION (donnees brutes) ===")
    print("Dimensions :", df.shape)
    print("Colonnes   :", list(df.columns))
    print("Types      :\n", df.dtypes)
    print("Valeurs manquantes :\n", df.isna().sum())
    print("Lignes dupliquees  :", int(df.duplicated().sum()))
    print("Apercu :\n", df.head())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    text_col = find_col(df, TEXT_CANDIDATES)
    label_col = find_col(df, LABEL_CANDIDATES)
    type_col = find_col(df, TYPE_CANDIDATES)
    if text_col is None or label_col is None:
        raise SystemExit(
            f"Colonnes texte/label introuvables. Colonnes presentes : {list(df.columns)}. "
            "Adaptez TEXT_CANDIDATES / LABEL_CANDIDATES dans app/cleaning.py."
        )
    print(f"\nColonnes utilisees : texte='{text_col}', label='{label_col}', type='{type_col}'")
    print("Valeurs de label distinctes :", sorted(df[label_col].dropna().astype(str).unique())[:20])

    n0 = len(df)
    out = pd.DataFrame(
        {
            "text_raw": df[text_col],
            "label_raw": df[label_col],
            "type": df[type_col] if type_col else "inconnu",
        }
    )

    out = out.dropna(subset=["text_raw"])
    n1 = len(out)
    out["text"] = out["text_raw"].map(clean_text)
    out = out[out["text"] != ""]
    n2 = len(out)

    out["label"] = out["label_raw"].map(normalize_label)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    n3 = len(out)

    # Textes identiques avec labels contradictoires -> ambigus, on les retire
    nunique = out.groupby("text")["label"].nunique()
    conflicts = nunique[nunique > 1].index
    out = out[~out["text"].isin(conflicts)]
    n4 = len(out)
    out = out.drop_duplicates(subset="text")
    n5 = len(out)

    out["type"] = out["type"].astype(str).str.strip().str.lower()
    out.loc[out["type"].isin(["", "nan", "none"]), "type"] = "inconnu"
    out["label_name"] = out["label"].map({1: "harcelement", 0: "non_harcelement"})
    out["n_words"] = out["text"].str.split().str.len()
    out["n_chars"] = out["text"].str.len()

    print("\n=== RAPPORT DE NETTOYAGE ===")
    print(f"Lignes initiales                      : {n0}")
    print(f"- textes manquants                    : {n0 - n1}")
    print(f"- textes vides apres nettoyage        : {n1 - n2}")
    print(f"- labels manquants                    : {n2 - n3}")
    print(f"- textes ambigus (labels contradictoires) : {n3 - n4}")
    print(f"- doublons                            : {n4 - n5}")
    print(f"Lignes finales                        : {n5}")
    print("\nRepartition des classes :\n", out["label_name"].value_counts())
    print("\nRepartition des types :\n", out["type"].value_counts())

    cols = ["text_raw", "text", "label", "label_name", "type", "n_words", "n_chars"]
    return out[cols].reset_index(drop=True)


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"{RAW_CSV} introuvable. Lancez d'abord : python download_data.py")
    raw = read_raw()
    explore(raw)
    df = clean(raw)
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"\nFichier ecrit : {CLEAN_CSV}")


if __name__ == "__main__":
    main()
