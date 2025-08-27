from __future__ import annotations
import numpy as np
import pandas as pd
from config import RATING_ORDER, TARGET_SHARES, OVERLAY_CAPS

# ---------- utilitaires ----------
def _normtxt(s: str) -> str:
    s = str(s)
    for a, b in (("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
                 ("à", "a"), ("â", "a"),
                 ("î", "i"), ("ï", "i"),
                 ("ô", "o"),
                 ("ù", "u"), ("ü", "u"),
                 ("ç", "c")):
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s.strip().lower()

def r_idx(r: str) -> int:
    return RATING_ORDER.index(r) if r in RATING_ORDER else len(RATING_ORDER) - 1

def shares_to_quantiles(shares: dict[str, float]) -> dict[str, float]:
    cum, out = 0.0, {}
    for r in RATING_ORDER:
        cum += shares.get(r, 0) / 100.0
        out[r] = max(0.0, min(1.0, cum))
    return out

Q_TARGET = shares_to_quantiles(TARGET_SHARES)

# ---------- absolu dynamique ----------
def _quantile_edges_from_pd(pd_values: pd.Series) -> dict:
    q_targets = [Q_TARGET[r] for r in RATING_ORDER]
    quantiles = pd_values.quantile(q_targets).values
    edges = {r: float(q) for r, q in zip(RATING_ORDER, quantiles)}
    edges[RATING_ORDER[-1]] = 1.0
    return edges

def prob_to_abs_rating_dynamic(p: float, edges: dict) -> str:
    for i, r in enumerate(RATING_ORDER):
        lo = -np.inf if i == 0 else edges[RATING_ORDER[i - 1]]
        hi = edges[r]
        if lo < p <= hi:
            return r
    return "C"

# ---------- overlay secteur + cap ----------
def sector_bonus_value(sector: str | None) -> int:
    if sector is None or (isinstance(sector, float) and np.isnan(sector)):
        return 0
    s = _normtxt(sector)
    if any(k in s for k in ["eau", "electric", "energie", "utility", "electricite", "électricité"]):
        return 2
    if any(k in s for k in ["telecom", "télécom"]):
        return 2
    if any(k in s for k in ["banque", "bank", "finance", "assur"]):
        return 1
    if any(k in s for k in ["industrie", "manufact", "services", "transport", "logist", "agro", "mines"]):
        return 1
    return 0

def apply_sector_overlay(note_base: str, pdv: float, sector: str | None) -> tuple[str, int]:
    max_b = sector_bonus_value(sector)
    if pdv >= OVERLAY_CAPS["no_bonus_if_pd_ge"]:
        allowed = 0
    elif pdv >= OVERLAY_CAPS["max_bonus_if_pd_ge"]:
        allowed = min(max_b, OVERLAY_CAPS["max_bonus_mid"])
    else:
        allowed = max_b
    new_idx = max(0, r_idx(note_base) - allowed)
    return RATING_ORDER[new_idx], allowed

def cap_vs_absolute(note_overlay: str, note_abs: str) -> str:
    i_abs, i_ov = r_idx(note_abs), r_idx(note_overlay)
    i_allow = max(0, i_abs - OVERLAY_CAPS["max_up_over_abs"])
    return RATING_ORDER[i_allow] if i_ov < i_allow else note_overlay

# ---------- mélange ----------
def _blend_notes(a: str, b: str) -> str:
    ia, ib = r_idx(a), r_idx(b)
    im = int(round((ia + ib) / 2))
    im = max(0, min(im, len(RATING_ORDER) - 1))
    return RATING_ORDER[im]

# ---------- pipeline de notation ----------
def apply_full_notation(df: pd.DataFrame,
                        col_pd: str,
                        col_year: str,
                        col_sector: str) -> pd.DataFrame:
    out = df.copy()
    out[col_pd] = pd.to_numeric(out[col_pd], errors="coerce").clip(0, 1)
    out = out.dropna(subset=[col_pd]).copy()
    if out.empty:
        return out

    # seuils dynamiques globaux
    dyn_edges = _quantile_edges_from_pd(out[col_pd])

    # quantiles par annee pour prudence
    out["__u__"] = out.groupby(col_year)[col_pd].rank(pct=True, method="average")

    # notes intermediaires
    out["Notation_absolue"] = out[col_pd].apply(lambda x: prob_to_abs_rating_dynamic(x, dyn_edges))
    out["Notation_quantiles"] = out["__u__"].apply(
        lambda u: next((r for r in RATING_ORDER if u <= Q_TARGET[r]), "C")
    )

    # blend
    out["Notation_prudente"] = [
        _blend_notes(a, b) for a, b in zip(out["Notation_quantiles"], out["Notation_absolue"])
    ]

    # overlay + cap
    notes_ov, bonuses = [], []
    for nb, pdv, sect, na in zip(out["Notation_prudente"], out[col_pd], out[col_sector], out["Notation_absolue"]):
        no, b = apply_sector_overlay(nb, pdv, sect)
        no = cap_vs_absolute(no, na)
        notes_ov.append(no)
        bonuses.append(b)

    out["Notation_overlay"] = notes_ov
    out["Overlay_bonus"] = bonuses
    out["Notation_finale"] = out["Notation_overlay"]
    out["Reason"] = [
        f"ABS={na} | Q={nq} | B+{b}"
        for na, nq, b in zip(out["Notation_absolue"], out["Notation_quantiles"], out["Overlay_bonus"])
    ]

    return out.drop(columns=["__u__"])
