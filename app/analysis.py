"""Etape 3 : analyse descriptive (lecture depuis MongoDB, sinon CSV)."""
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from app import visualization as viz
from app.config import REPORTS_DIR
from app.db import load_dataset


def top_words_by_class(df: pd.DataFrame, n: int = 20) -> dict:
    out = {}
    for name, g in df.groupby("label_name"):
        vec = CountVectorizer(stop_words="english", max_features=5000)
        X = vec.fit_transform(g["text"])
        counts = Counter(dict(zip(vec.get_feature_names_out(), X.sum(axis=0).A1)))
        out[name] = counts.most_common(n)
    return out


def main() -> None:
    df = load_dataset()
    REPORTS_DIR.mkdir(exist_ok=True)
    lines = ["# Analyse descriptive", ""]

    def section(title, content):
        print(f"\n=== {title} ===\n{content}")
        lines.extend([f"## {title}", "", "```", str(content), "```", ""])

    section("Dimensions", f"{len(df)} messages, colonnes : {list(df.columns)}")
    balance = df["label_name"].value_counts()
    section("Repartition des classes", pd.concat([balance, (balance / len(df)).round(4)], axis=1,
                                                 keys=["effectif", "proportion"]))
    section("Longueur des messages par classe (mots)", df.groupby("label_name")["n_words"].describe().round(2))

    if df["type"].nunique() > 1:
        t = df.groupby("type")["label"].agg(effectif="count", taux_harcelement="mean").round(3)
        section("Types de messages et taux de harcelement", t.sort_values("effectif", ascending=False))

    top = top_words_by_class(df)
    section("Mots les plus frequents par classe",
            "\n".join(f"{k}: " + ", ".join(f"{w}({c})" for w, c in v[:15]) for k, v in top.items()))

    (REPORTS_DIR / "analyse_descriptive.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nFigures :")
    viz.plot_class_distribution(df)
    viz.plot_length_distribution(df)
    viz.plot_top_words(top)
    viz.plot_types(df)
    print("\nRapport : reports/analyse_descriptive.md")


if __name__ == "__main__":
    main()
