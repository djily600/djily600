# BRVM Risk — Application Flask (Upload Excel → PD → Notation AAA..C)

## Démarrer en local
```bash
conda create -n brvm-risk python=3.11 -y
conda activate brvm-risk
pip install -r requirements.txt
export FLASK_ENV=development
python app.py  # via 'flask run' si vous préférez
```
Puis ouvrez http://127.0.0.1:5000

## Déploiement Render
- `requirements.txt`, `Procfile`, `render.yaml` fournis.
- Ajouter des modèles dans `models/` (classifier.joblib, pipeline.joblib) si vous voulez utiliser l'IA complète.
- Sinon, fournissez un Excel avec la colonne `Proba_defaillance` pour un mapping direct PD→note.
