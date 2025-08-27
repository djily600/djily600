# services/labeling.py
from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULTS = {
    "levier_excessif": 1.0,   # Levier financier > 1.0 => critère KO
    "surendettement": 1.5,    # total dettes / capitaux propres > 1.5 => KO
}

def _norm_cols(df: pd.DataFrame) -> dict:
    """map 'nom_normalise' -> 'Nom original' (lower + strip)"""
    return {str(c).strip().lower(): c for c in df.columns}

def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    r = num.astype(float) / den.replace(0, np.nan).astype(float)
    return r.replace([np.inf, -np.inf], np.nan)

def compute_defaillance(df: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """
    Règles de défaillance (1) si AU MOINS un critère est vrai:
      - Bénéfice net < 0
      - EBE < 0
      - Capitaux propres ≤ 0
      - ROE < 0
      - total dettes / capitaux propres > 1.5
      - Levier financier > 1.0
      - Fonds de roulement < 0
    Sinon 0 (sain).
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    col = _norm_cols(df)

    def has(name: str) -> bool:
        return name in col

    bn  = col.get("bénéfice net")
    ebe = col.get("ebe")
    cp  = col.get("capitaux propres")
    fr  = col.get("fonds de roulement")
    td  = col.get("total dettes")
    roe = col.get("rendement des capitaux propres (roe)")
    lev = col.get("levier financier")

    crits = []

    if bn and bn in df:   crits.append(df[bn].astype(float) < 0)
    if ebe and ebe in df: crits.append(df[ebe].astype(float) < 0)
    if cp and cp in df:   crits.append(df[cp].astype(float) <= 0)
    if roe and roe in df: crits.append(df[roe].astype(float) < 0)
    if td and cp and td in df and cp in df:
        ratio_dc = _safe_ratio(df[td], df[cp])
        crits.append(ratio_dc > cfg["surendettement"])
    if lev and lev in df: crits.append(df[lev].astype(float) > cfg["levier_excessif"])
    if fr and fr in df:   crits.append(df[fr].astype(float) < 0)

    if not crits:
        # Si aucune colonne attendue n'est trouvée, on renvoie 0 (sain)
        return pd.Series(0, index=df.index, name="Défaillance").astype(int)

    any_bad = crits[0]
    for k in crits[1:]:
        any_bad = any_bad | k

    return any_bad.astype(int).rename("Défaillance")
