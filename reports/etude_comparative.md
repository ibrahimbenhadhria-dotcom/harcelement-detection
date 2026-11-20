# Etude comparative des deux modeles de prediction

## 1. Contexte

- Jeu de donnees nettoye : 5738 messages (4590 entrainement / 1148 test, decoupage stratifie).
- Repartition des classes : harcelement = 54.7%, non_harcelement = 45.3%.
- Representation du texte : TF-IDF (unigrammes + bigrammes, min_df=2, sublinear_tf).
- Hyperparametres choisis par GridSearchCV (3 plis, score F1) sur le jeu d'entrainement uniquement.

## 2. Modeles compares

- **Regression logistique** : modele lineaire discriminant, `class_weight='balanced'`. Interpretable (coefficients par mot).
- **Naive Bayes multinomial** : modele probabiliste generatif, tres rapide, suppose l'independance des mots.

## 3. Resultats sur le jeu de test

| Modele | Accuracy | Precision | Recall | F1 | ROC-AUC | Temps recherche+entrainement (s) | Prediction (ms/msg) |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.9042 | 0.8948 | 0.9347 | 0.9143 | 0.9625 | 3.1 | 0.015 |
| MultinomialNB | 0.9068 | 0.8953 | 0.9395 | 0.9169 | 0.9705 | 2.5 | 0.009 |

Meilleurs hyperparametres : LogisticRegression -> {'clf__C': 5}; MultinomialNB -> {'clf__alpha': 0.1}

## 4. Matrices de confusion

Lignes = reel, colonnes = predit (non_harcelement, harcelement).

- LogisticRegression : VN=451, FP=69, FN=41, VP=587
- MultinomialNB : VN=451, FP=69, FN=38, VP=590

## 5. Analyse

- **MultinomialNB** obtient le meilleur F1 (0.9169 contre 0.9143 pour LogisticRegression, ecart +0.0025).
- Meilleur rappel (moins de harcelements rates) : **MultinomialNB**. Meilleure precision (moins de fausses alertes) : **MultinomialNB**.
- Le plus rapide a entrainer : **MultinomialNB**.
- Le F1 est le critere de selection car il equilibre precision et rappel, ce qui convient si les classes sont desequilibrees.

## 6. Conclusion et limites

Le modele retenu pour l'API est **MultinomialNB** (`models/best_model.joblib`).

Limites : texte traite comme un sac de mots (pas de contexte, d'ironie ni de sarcasme) ; nettoyage prevu pour l'anglais ; un seul decoupage train/test ; des modeles de type transformer (BERT) pourraient faire mieux mais coutent plus cher.

## 7. Figures

- `figures/05_matrices_confusion.png`
- `figures/06_comparaison_metriques.png`
- `figures/07_courbes_roc.png`
- `figures/08_facteurs_influents.png`