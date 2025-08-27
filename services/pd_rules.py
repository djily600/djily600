# services/pd_rules.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _norm_cols(df: pd.DataFrame) -> dict:
    return {str(c).strip().lower(): c for c in df.columns}

def pd_from_rules(df: pd.DataFrame) -> pd.Series:
    """
    PD simple 0..1 basée sur le nombre de critères KO.
    Dès que tu ajoutes ton vrai modèle, celui-ci prendra le relais.
    """
    col = _norm_cols(df)
    def get(name): return col.get(name)

    signals = []

    bn  = get("bénéfice net")
    ebe = get("ebe")
    cp  = get("capitaux propres")
    fr  = get("fonds de roulement")
    td  = get("total dettes")
    roe = get("rendement des capitaux propres (roe)")
    lev = get("levier financier")

    def s(series, cond):
        return cond(series.astype(float)).astype(int)

    if bn:  signals.append(s(df[bn],  lambda s: s < 0))
    if ebe: signals.append(s(df[ebe], lambda s: s < 0))
    if cp:  signals.append(s(df[cp],  lambda s: s <= 0))
    if roe: signals.append(s(df[roe], lambda s: s < 0))
    if td and cp:
        ratio = df[td].astype(float) / df[cp].replace(0, np.nan).astype(float)
        signals.append((ratio > 1.5).fillna(0).astype(int))
    if lev: signals.append(s(df[lev], lambda s: s > 1.0))
    if fr:  signals.append(s(df[fr],  lambda s: s < 0))

    if not signals:
        return pd.Series(0.05, index=df.index, name="Proba_defaillance")  # valeur par défaut

    bad_count = sum(signals)
    pd_est = (bad_count / 5.0).clip(0, 1)
    return pd_est.rename("Proba_defaillance").astype(float)
