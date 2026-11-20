"""Etapes 4 et 5 : deux modeles de prediction + etude comparative."""
import json
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app import visualization as viz
from app.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from app.db import load_dataset


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)


def build_candidates():
    return {
        "LogisticRegression": (
            Pipeline([("tfidf", tfidf()),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
            {"clf__C": [0.5, 1, 5]},
        ),
        "MultinomialNB": (
            Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
    }


def evaluate(name, pipe, search_time, X_test, y_test):
    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    pred_time = (time.perf_counter() - t0) / max(len(X_test), 1) * 1000
    proba = pipe.predict_proba(X_test)[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, proba)
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "train_time_s": float(search_time),
            "predict_ms_per_message": float(pred_time),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "report": classification_report(y_test, y_pred, target_names=["non_harcelement", "harcelement"],
                                        zero_division=0),
    }


def write_study(results, best, n_train, n_test, balance):
    names = list(results)
    L = ["# Etude comparative des deux modeles de prediction", ""]
    L += ["## 1. Contexte", "",
          f"- Jeu de donnees nettoye : {n_train + n_test} messages "
          f"({n_train} entrainement / {n_test} test, decoupage stratifie).",
          "- Repartition des classes : " + ", ".join(f"{k} = {v:.1%}" for k, v in balance.items()) + ".",
          "- Representation du texte : TF-IDF (unigrammes + bigrammes, min_df=2, sublinear_tf).",
          "- Hyperparametres choisis par GridSearchCV (3 plis, score F1) sur le jeu d'entrainement uniquement.", ""]
    L += ["## 2. Modeles compares", "",
          "- **Regression logistique** : modele lineaire discriminant, `class_weight='balanced'`. "
          "Interpretable (coefficients par mot).",
          "- **Naive Bayes multinomial** : modele probabiliste generatif, tres rapide, "
          "suppose l'independance des mots.", ""]
    L += ["## 3. Resultats sur le jeu de test", "",
          "| Modele | Accuracy | Precision | Recall | F1 | ROC-AUC | Temps recherche+entrainement (s) | Prediction (ms/msg) |",
          "|---|---|---|---|---|---|---|---|"]
    for n in names:
        m = results[n]["metrics"]
        L.append(f"| {n} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                 f"{m['roc_auc']:.4f} | {m['train_time_s']:.1f} | {m['predict_ms_per_message']:.3f} |")
    L += ["", "Meilleurs hyperparametres : " +
          "; ".join(f"{n} -> {results[n]['best_params']}" for n in names), ""]
    L += ["## 4. Matrices de confusion", "", "Lignes = reel, colonnes = predit (non_harcelement, harcelement).", ""]
    for n in names:
        cm = results[n]["confusion_matrix"]
        L.append(f"- {n} : VN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, VP={cm[1][1]}")
    L += ["", "## 5. Analyse", ""]
    a, b = names
    ma, mb = results[a]["metrics"], results[b]["metrics"]
    other = b if best == a else a
    mbest, mother = results[best]["metrics"], results[other]["metrics"]
    L.append(f"- **{best}** obtient le meilleur F1 ({mbest['f1']:.4f} contre {mother['f1']:.4f} "
             f"pour {other}, ecart {mbest['f1'] - mother['f1']:+.4f}).")
    rec_win = a if ma["recall"] >= mb["recall"] else b
    pre_win = a if ma["precision"] >= mb["precision"] else b
    L.append(f"- Meilleur rappel (moins de harcelements rates) : **{rec_win}**. "
             f"Meilleure precision (moins de fausses alertes) : **{pre_win}**.")
    fast = a if ma["train_time_s"] <= mb["train_time_s"] else b
    L.append(f"- Le plus rapide a entrainer : **{fast}**.")
    L.append("- Le F1 est le critere de selection car il equilibre precision et rappel, "
             "ce qui convient si les classes sont desequilibrees.")
    L += ["", "## 6. Conclusion et limites", "",
          f"Le modele retenu pour l'API est **{best}** (`models/best_model.joblib`).",
          "",
          "Limites : texte traite comme un sac de mots (pas de contexte, d'ironie ni de sarcasme) ; "
          "nettoyage prevu pour l'anglais ; un seul decoupage train/test ; "
          "des modeles de type transformer (BERT) pourraient faire mieux mais coutent plus cher.",
          "", "## 7. Figures", "",
          "- `figures/05_matrices_confusion.png`", "- `figures/06_comparaison_metriques.png`",
          "- `figures/07_courbes_roc.png`", "- `figures/08_facteurs_influents.png`"]
    (REPORTS_DIR / "etude_comparative.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE)
    print(f"Train : {len(X_train)}  Test : {len(X_test)}")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    results = {}
    for name, (pipe, grid) in build_candidates().items():
        print(f"\n--- Entrainement : {name} ---")
        search = GridSearchCV(pipe, grid, cv=3, scoring="f1", n_jobs=-1)
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        search_time = time.perf_counter() - t0
        res = evaluate(name, search.best_estimator_, search_time, X_test, y_test)
        res["best_params"] = search.best_params_
        res["pipeline"] = search.best_estimator_
        results[name] = res
        print("Meilleurs parametres :", search.best_params_)
        print(res["report"])
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")

    best = max(results, key=lambda n: (results[n]["metrics"]["f1"], results[n]["metrics"]["roc_auc"]))
    print(f"\nMeilleur modele : {best}")
    joblib.dump(results[best]["pipeline"], MODELS_DIR / "best_model.joblib")

    summary = {"best_model": best, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
               "models": {n: {"metrics": r["metrics"], "best_params": {k: float(v) for k, v in r["best_params"].items()}}
                          for n, r in results.items()}}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    balance = df["label_name"].value_counts(normalize=True)
    write_study(results, best, len(X_train), len(X_test), balance)

    print("\nFigures :")
    viz.plot_confusion_matrices(results)
    viz.plot_metrics_comparison(results)
    viz.plot_roc(results)
    if "LogisticRegression" in results:
        viz.plot_top_features(results["LogisticRegression"]["pipeline"])
    print("\nEtude comparative : reports/etude_comparative.md")


if __name__ == "__main__":
    main()
