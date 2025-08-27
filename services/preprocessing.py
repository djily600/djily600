from __future__ import annotations
import pandas as pd
import numpy as np

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de base : convertir les colonnes texte avec virgule en float."""
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].str.replace(",", ".").astype(float)
        except Exception:
            pass
    return df


def squash_pd(pd_series: pd.Series) -> pd.Series:
    """Compression légère des proba pour éviter qu’elles soient toutes tassées vers 0 ou 1."""
    eps = 1e-6
    p = pd_series.astype(float).clip(eps, 1 - eps)
    logit = np.log(p / (1 - p))
    logit2 = logit / 1.5   # augmente à 2.0 si tu veux plus lisser
    p2 = 1.0 / (1.0 + np.exp(-logit2))
    return pd.Series(p2, index=pd_series.index, name=pd_series.name)
