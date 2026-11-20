"""Etape 6 : visualisation des resultats (figures sauvegardees dans reports/figures)."""
import matplotlib

matplotlib.use("Agg")  # pas de fenetre, on sauvegarde en PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from app.config import FIG_DIR  # noqa: E402

sns.set_theme(style="whitegrid")
COLORS = {"harcelement": "#d62728", "non_harcelement": "#1f77b4"}
CLASS_NAMES = ["non_harcelement", "harcelement"]


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)
    print("  figure :", name)


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label_name"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[COLORS.get(i, "gray") for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Repartition des classes")
    ax.set_ylabel("Nombre de messages")
    ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, "01_repartition_classes.png")


def plot_length_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cap = df["n_words"].quantile(0.99)
    for name, g in df.groupby("label_name"):
        ax.hist(g["n_words"].clip(upper=cap), bins=40, alpha=0.6, label=name, color=COLORS.get(name))
    ax.set_title("Distribution de la longueur des messages (mots)")
    ax.set_xlabel("Nombre de mots")
    ax.set_ylabel("Frequence")
    ax.legend()
    _save(fig, "02_longueur_messages.png")


def plot_top_words(top: dict, n: int = 15) -> None:
    fig, axes = plt.subplots(1, len(top), figsize=(6 * len(top), 6))
    axes = np.atleast_1d(axes)
    for ax, (name, words) in zip(axes, top.items()):
        words = words[:n][::-1]
        ax.barh([w for w, _ in words], [c for _, c in words], color=COLORS.get(name))
        ax.set_title(f"Mots les plus frequents : {name}")
    _save(fig, "03_mots_frequents.png")


def plot_types(df: pd.DataFrame) -> None:
    if df["type"].nunique() <= 1:
        return
    ct = pd.crosstab(df["type"], df["label_name"])
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(ct))))
    ct.plot(kind="barh", stacked=True, ax=ax, color=[COLORS.get(c) for c in ct.columns])
    ax.set_title("Messages par type et par classe")
    ax.set_xlabel("Nombre de messages")
    _save(fig, "04_types_harcelement.png")


def plot_confusion_matrices(results: dict) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"Matrice de confusion : {name}")
        ax.set_xlabel("Predit")
        ax.set_ylabel("Reel")
    _save(fig, "05_matrices_confusion.png")


def plot_metrics_comparison(results: dict) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    w = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, r) in enumerate(results.items()):
        vals = [r["metrics"][m] for m in metrics]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title("Comparaison des performances (jeu de test)")
    ax.legend()
    _save(fig, "06_comparaison_metriques.png")


def plot_roc(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC")
    ax.legend(loc="lower right")
    _save(fig, "07_courbes_roc.png")


def plot_top_features(pipeline, n: int = 20) -> None:
    """Facteurs les plus influents (coefficients de la regression logistique)."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return
    names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    idx = np.concatenate([order[:n], order[-n:]])
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [COLORS["harcelement"] if c > 0 else COLORS["non_harcelement"] for c in coefs[idx]]
    ax.barh(names[idx], coefs[idx], color=colors)
    ax.set_title("Mots/expressions les plus associes a chaque classe\n(rouge = harcelement, bleu = non harcelement)")
    ax.set_xlabel("Coefficient (regression logistique)")
    _save(fig, "08_facteurs_influents.png")
