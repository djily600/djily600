# services/inference.py — robuste à tous les formats (pipeline complet, pipeline transform, ou pas de pipeline)
from __future__ import annotations
import os, json, joblib, pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLF_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
PIPE_PATH = os.path.join(MODEL_DIR, "pipeline.joblib")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.json")

class NoModelAvailable(Exception):
    pass

def _reorder_features_if_needed(df_features: pd.DataFrame) -> pd.DataFrame:
    """Optionnel : si feature_list.json présent, impose l'ordre/ajoute colonnes manquantes (=0)."""
    if os.path.exists(FEATURE_LIST_PATH):
        with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
            feat_list = json.load(f)
        for c in feat_list:
            if c not in df_features.columns:
                df_features[c] = 0.0
        df_features = df_features[feat_list]
    return df_features

def predict_pd(df_features: pd.DataFrame) -> pd.Series:
    """
    Prédit la probabilité de défaillance (classe 1) avec la logique suivante :
      A) pipeline.joblib existe et possède predict_proba -> on l'utilise directement (end-to-end)
      B) pipeline.joblib existe et possède transform -> on transforme, puis classifier.joblib fait predict_proba
      C) pas de pipeline -> classifier.joblib fait predict_proba sur les features numériques
    """
    # 0) features : imposer l'ordre si présent
    df_features = _reorder_features_if_needed(df_features)

    pipe = None
    clf = None

    has_pipe = os.path.exists(PIPE_PATH)
    has_clf  = os.path.exists(CLF_PATH)

    if not has_pipe and not has_clf:
        raise NoModelAvailable("Aucun modèle n'est disponible dans /models (ni pipeline.joblib ni classifier.joblib).")

    if has_pipe:
        pipe = joblib.load(PIPE_PATH)
        # Cas A : pipeline a predict_proba (pipeline complet)
        if hasattr(pipe, "predict_proba"):
            pd_pred = pipe.predict_proba(df_features)[:, 1]
            return pd.Series(pd_pred, index=df_features.index, name="Proba_defaillance")

    # Si on arrive ici, soit pas de pipe, soit pipe sans predict_proba.
    # On tente B : pipe.transform + clf.predict_proba
    if has_clf:
        clf = joblib.load(CLF_PATH)

    X = df_features
    if pipe is not None and hasattr(pipe, "transform"):
        X = pipe.transform(df_features)

    if clf is None:
        # On a un pipeline sans predict_proba et pas de classifier séparé -> pas possible
        raise NoModelAvailable("pipeline.joblib sans predict_proba et sans classifier.joblib.")

    # Cas B / C : classifier sur X
    if hasattr(clf, "predict_proba"):
        pd_pred = clf.predict_proba(X)[:, 1]
        return pd.Series(pd_pred, index=df_features.index, name="Proba_defaillance")

    # Dernier recours (peu probable) : pas de predict_proba -> on fabrique une proba à partir de la prédiction
    if hasattr(clf, "predict"):
        y_hat = clf.predict(X)
        # 0/1 -> 0.05 / 0.95 pour simuler une "proba"
        pd_pred = (y_hat.astype(float) * 0.9) + 0.05
        return pd.Series(pd_pred, index=df_features.index, name="Proba_defaillance")

    raise NoModelAvailable("Impossible de prédire : ni predict_proba ni predict disponible sur le modèle.")
