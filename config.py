import os

# Upload config
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {'.xlsx', '.xls'}

RATING_ORDER = ["AAA","AA","A","BBB","BB","B","CCC","CC","C"]

# Seuils absolus (plus permissifs)
ABS_EDGES = {
    "AAA": 0.02,
    "AA":  0.05,
    "A":   0.10,
    "BBB": 0.18,
    "BB":  0.28,
    "B":   0.40,
    "CCC": 0.55,
    "CC":  0.75,
    "C":   1.00
}

# Cible de parts par quantiles (tu peux augmenter AAA/AA si tu veux en voir plus)
TARGET_SHARES = {"AAA":7,"AA":12,"A":16,"BBB":20,"BB":18,"B":13,"CCC":8,"CC":4,"C":2}

# Overlay / caps (on autorise un peu plus de “bonus” secteur)
OVERLAY_CAPS = {
    "no_bonus_if_pd_ge": 0.70,  # avant 0.60 -> moins de blocage
    "max_bonus_if_pd_ge": 0.25,  # avant 0.20
    "max_bonus_mid": 2,
    "max_up_over_abs": 3        # avant 2 -> autorise +3 crans au-dessus de l'absolu
}


# Required columns if no model is available and we want direct PD mapping
MIN_COLS_DIRECT = {"Entreprise", "Secteur", "Proba_defaillance"}
