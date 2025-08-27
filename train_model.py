# ============================
# train_model.py
# RandomForest avec imputation + calibration (probabilités réalistes)
# ============================

import pandas as pd
import numpy as np
import json, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

# ==== Chemin vers TON Excel (.xlsx) ====
EXCEL_PATH = r"C:\Users\HP\OneDrive\Desktop\Revue_Litterature\EF_Entreprises_cotées.xlsx"
SHEET_NAME = 0   # ou le nom de l’onglet

# ==== Lecture des données ====
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME).copy()

# Nettoyage simple : convertir les "numériques-texte" (virgules -> points)
for col in df.select_dtypes(include=["object"]).columns:
    try:
        df[col] = df[col].str.replace(",", ".").astype(float)
    except Exception:
        pass

# ==== Calcul de la cible Défaillance (règles métier) ====
def compute_defaillance(df: pd.DataFrame) -> pd.Series:
    col = {str(c).strip().lower(): c for c in df.columns}
    def get(name): return col.get(name)

    bn  = get("bénéfice net")
    ebe = get("ebe")
    cp  = get("capitaux propres")
    fr  = get("fonds de roulement")
    td  = get("total dettes")
    roe = get("rendement des capitaux propres (roe)")
    lev = get("levier financier")

    crits = []
    if bn:  crits.append(df[bn].astype(float) < 0)
    if ebe: crits.append(df[ebe].astype(float) < 0)
    if cp:  crits.append(df[cp].astype(float) <= 0)
    if roe: crits.append(df[roe].astype(float) < 0)
    if td and cp:
        ratio_dc = df[td].astype(float) / df[cp].replace(0, np.nan).astype(float)
        crits.append(ratio_dc > 1.5)
    if lev: crits.append(df[lev].astype(float) > 1.0)
    if fr:  crits.append(df[fr].astype(float) < 0)

    if not crits:
        return pd.Series(0, index=df.index, name="Défaillance").astype(int)

    any_bad = crits[0]
    for c in crits[1:]:
        any_bad = any_bad | c
    return any_bad.astype(int).rename("Défaillance")

df["Défaillance"] = compute_defaillance(df)

# ==== Constitution X / y ====
drop_text_cols = ["NOM DE L'ENTREPRISE", "SECTEUR D'ACTIVITE", "PAYS", "ANNEE"]
X = df.drop(columns=[c for c in drop_text_cols + ["Défaillance"] if c in df.columns], errors="ignore") \
      .select_dtypes(include=["number"]).copy()
y = df["Défaillance"].astype(int)

# Remplace les +/-inf par NaN (sécurité)
X = X.replace([np.inf, -np.inf], np.nan)

# ==== Split train/test ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()==2 else None
)

# ==== Base RF ====
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced_subsample"   # <= gère le déséquilibre
)

# Pipeline: imputer -> RF
base_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", rf),
])

# Calibration des probabilités
calib = CalibratedClassifierCV(
    estimator=base_pipe,
    method="isotonic",   # ou "sigmoid" si peu d’exemples
    cv=5
)

calib.fit(X_train, y_train)

# ==== Évaluation ====
proba = calib.predict_proba(X_test)[:, 1]
pred  = (proba >= 0.5).astype(int)
print("Test AUC:", round(roc_auc_score(y_test, proba), 3))
print("Test Acc:", round(accuracy_score(y_test, pred), 3))

# ==== Sauvegardes pour l'application ====
joblib.dump(calib, "pipeline.joblib")
print("✅ pipeline.joblib écrit (calibré)")

joblib.dump(rf, "classifier.joblib")  # pas indispensable mais conservé
print("✅ classifier.joblib écrit")

with open("feature_list.json", "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, ensure_ascii=False, indent=2)
print("ℹ️ feature_list.json écrit")

# ==== Diagnostic des probabilités ====
import numpy as np
print("— DIAG —")
print("PD train (min/median/max):",
      round(float(calib.predict_proba(X_train)[:,1].min()), 4),
      round(float(np.median(calib.predict_proba(X_train)[:,1])), 4),
      round(float(calib.predict_proba(X_train)[:,1].max()), 4))
print("PD test  (min/median/max):",
      round(float(calib.predict_proba(X_test)[:,1].min()), 4),
      round(float(np.median(calib.predict_proba(X_test)[:,1])), 4),
      round(float(calib.predict_proba(X_test)[:,1].max()), 4))
