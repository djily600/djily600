from __future__ import annotations
import os, io, uuid
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import pandas as pd
from werkzeug.exceptions import RequestEntityTooLarge

# --- Config & services ---
from config import (
    MAX_CONTENT_LENGTH,
    ALLOWED_EXTENSIONS,
    RATING_ORDER,
)
from services.io_excel import read_excel
from services.preprocessing import basic_clean, squash_pd
from services.inference import predict_pd, NoModelAvailable
from services.labeling import compute_defaillance
from services.pd_rules import pd_from_rules
from services.rating import apply_full_notation  # seuils dynamiques, overlay, cap

TARGET_COMPANY = "SONATEL SENEGAL"  # filtrage par defaut

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("APP_SECRET_KEY", "dev-secret")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # Memoire volatile (prod: cache/DB)
    RESULTS: dict[str, pd.DataFrame] = {}

    # ---------------- VUES ----------------
    @app.route("/", methods=["GET"])
    def home():
        return render_template("upload.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        # 0) Fichier
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Veuillez sélectionner un fichier Excel (.xlsx/.xls).")
            return redirect(url_for("home"))
        if not allowed_file(file.filename):
            flash("Format non autorisé. Formats acceptés : .xlsx, .xls")
            return redirect(url_for("home"))

        # 1) Lecture + nettoyage
        stream = io.BytesIO(file.read())
        try:
            df = read_excel(stream)
        except Exception as e:
            flash(f"Impossible de lire le fichier Excel : {e}")
            return redirect(url_for("home"))

        df = basic_clean(df)

        # 2) Cible metier
        df["Défaillance"] = compute_defaillance(df)

        # 3) PD modele si dispo, sinon regles
        try:
            id_cols = [c for c in df.columns if c.lower() in {
                "nom de l'entreprise", "secteur d'activite", "secteur",
                "identifiant", "annee", "pays"
            }]
            drop_cols = set(id_cols + ["Défaillance"])
            features = (
                df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                  .select_dtypes(include=["number"])
            )
            raw_pd = predict_pd(features)
        except NoModelAvailable:
            raw_pd = pd_from_rules(df)

        pd_adj = squash_pd(raw_pd)  # lissage leger

        # 4) Notation complete
        result = df.copy()
        result["Proba_defaillance"] = pd_adj.values

        # Colonnes robustes ANNEE / SECTEUR
        col_pd = "Proba_defaillance"

        year_candidates = [c for c in result.columns if c.upper().strip() == "ANNEE"]
        if year_candidates:
            col_year = year_candidates[0]
        else:
            col_year = "__ANNEE__"
            result[col_year] = ""

        sector_candidates = [c for c in result.columns if c.upper().strip() in {"SECTEUR D'ACTIVITE", "SECTEUR"}]
        if sector_candidates:
            col_sector = sector_candidates[0]
        else:
            col_sector = "__SECTEUR__"
            result[col_sector] = "Inconnu"

        result = apply_full_notation(result, col_pd=col_pd, col_year=col_year, col_sector=col_sector)

        # 5) Statut simple
        if "Défaillance" in result.columns:
            result["Statut"] = result["Défaillance"].map({1: "Défaillante", 0: "Saine"}).fillna("Inconnu")
        else:
            result["Statut"] = (result["Proba_defaillance"] >= 0.5).map({True: "Défaillante", False: "Saine"})

        # 6) Stockage et redirection (filtre par defaut SONATEL SENEGAL)
        ticket = str(uuid.uuid4())
        RESULTS[ticket] = result
        return redirect(url_for("status", id=ticket, company=TARGET_COMPANY))

    @app.route("/status", methods=["GET"])
    def status():
        """Vue 1 — Entreprise | Année | Statut | PD (%) + bouton Evaluation note."""
        ticket = request.args.get("id")
        company = request.args.get("company", "").strip()
        if not ticket or ticket not in RESULTS:
            flash("Résultat introuvable.")
            return redirect(url_for("home"))

        df = RESULTS[ticket].copy()

        # Colonne entreprise
        ent_col = None
        for c in ["NOM DE L'ENTREPRISE", "Entreprise", "ENTREPRISE", "NOM ENTREPRISE", "NOM"]:
            if c in df.columns:
                ent_col = c
                break
        if not ent_col:
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            ent_col = text_cols[0] if text_cols else df.columns[0]

        # Colonne ANNEE robuste
        year_col = next((c for c in df.columns if c.upper().strip() == "ANNEE"), None)
        if not year_col:
            year_col = "__ANNEE__"
            df[year_col] = ""

        # Filtre entreprise optionnel
        if company:
            def _norm(s: str) -> str:
                s = str(s)
                rep = (("é","e"),("è","e"),("ê","e"),("ë","e"),
                       ("á","a"),("à","a"),("â","a"),("ä","a"),
                       ("í","i"),("ì","i"),("î","i"),("ï","i"),
                       ("ó","o"),("ò","o"),("ô","o"),("ö","o"),
                       ("ú","u"),("ù","u"),("û","u"),("ü","u"),
                       ("ç","c"))
            # apply replacements
                for a,b in rep:
                    s = s.replace(a,b).replace(a.upper(), b.upper())
                return " ".join(s.upper().split())

            target = _norm(company)
            ent_norm = df[ent_col].astype(str).map(_norm)
            mask = pd.Series(True, index=df.index)
            for t in target.split():
                mask &= ent_norm.str.contains(rf"\b{t}\b", regex=True, na=False)
            df = df[mask].copy()

        # Tableau minimal avec ANNEE
        cols = [ent_col, year_col, "Statut"]
        if "Proba_defaillance" in df.columns:
            df["_PD_percent"] = (df["Proba_defaillance"] * 100).round(2)
            cols.append("_PD_percent")

        table = (
            df[cols]
              .rename(columns={
                  ent_col: "Entreprise",
                  year_col: "Année",
                  "_PD_percent": "PD (%)",
              })
              .to_dict(orient="records")
        )

        # KPIs
        kpi = {
            "n": int(df.shape[0]),
            "nb_saines": int((df["Statut"] == "Saine").sum()) if "Statut" in df else 0,
            "nb_def": int((df["Statut"] == "Défaillante").sum()) if "Statut" in df else 0,
        }

        return render_template("status.html",
                               table=table, kpi=kpi, ticket=ticket, company=company)

    @app.route("/rating", methods=["GET"])
    def rating():
        """Vue 2 — Entreprise | Année | Notation finale."""
        ticket = request.args.get("id")
        company = request.args.get("company", "").strip()
        if not ticket or ticket not in RESULTS:
            flash("Résultat introuvable.")
            return redirect(url_for("home"))

        df = RESULTS[ticket].copy()

        # Colonne entreprise
        ent_col = None
        for c in ["NOM DE L'ENTREPRISE", "Entreprise", "ENTREPRISE", "NOM ENTREPRISE", "NOM"]:
            if c in df.columns:
                ent_col = c
                break
        if not ent_col:
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            ent_col = text_cols[0] if text_cols else df.columns[0]

        # Colonne ANNEE robuste
        year_col = next((c for c in df.columns if c.upper().strip() == "ANNEE"), None)
        if not year_col:
            year_col = "__ANNEE__"
            df[year_col] = ""

        # Filtre optionnel
        if company:
            def _norm(s: str) -> str:
                s = str(s)
                rep = (("é","e"),("è","e"),("ê","e"),("ë","e"),
                       ("á","a"),("à","a"),("â","a"),("ä","a"),
                       ("í","i"),("ì","i"),("î","i"),("ï","i"),
                       ("ó","o"),("ò","o"),("ô","o"),("ö","o"),
                       ("ú","u"),("ù","u"),("û","u"),("ü","u"),
                       ("ç","c"))
                for a,b in rep:
                    s = s.replace(a,b).replace(a.upper(), b.upper())
                return " ".join(s.upper().split())
            target = _norm(company)
            ent_norm = df[ent_col].astype(str).map(_norm)
            mask = pd.Series(True, index=df.index)
            for t in target.split():
                mask &= ent_norm.str.contains(rf"\b{t}\b", regex=True, na=False)
            df = df[mask].copy()

        if "Notation_finale" not in df.columns:
            flash("La note n'est pas disponible.")
            return redirect(url_for("status", id=ticket, company=company or None))

        table = (
            df[[ent_col, year_col, "Notation_finale"]]
              .rename(columns={ent_col: "Entreprise", year_col: "Année"})
              .to_dict(orient="records")
        )

        dist = (
            df["Notation_finale"].value_counts()
              .reindex(RATING_ORDER, fill_value=0)
              .to_dict()
        )

        return render_template("rating.html",
                               table=table, dist=dist, ticket=ticket, company=company)

    @app.route("/download", methods=["GET"])
    def download():
        ticket = request.args.get("id")
        fmt = request.args.get("fmt", "xlsx")
        if not ticket or ticket not in RESULTS:
            flash("Résultat introuvable.")
            return redirect(url_for("home"))
        df = RESULTS[ticket]

        if fmt == "csv":
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            return send_file(
                io.BytesIO(buf.getvalue().encode("utf-8")),
                as_attachment=True,
                download_name="notes.csv",
                mimetype="text/csv",
            )
        else:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Résultats")
            buf.seek(0)
            return send_file(
                buf,
                as_attachment=True,
                download_name="notes.xlsx",
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(e):
        flash(f"Votre fichier dépasse la limite de {MAX_CONTENT_LENGTH // (1024*1024)} Mo. "
              f"Réduisez la taille ou augmentez MAX_UPLOAD_MB.")
        return redirect(url_for("home"))

    return app

# WSGI
app = create_app()
