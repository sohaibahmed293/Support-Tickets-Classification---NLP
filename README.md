# Customer Support Ticket Classifier

Automated NLP pipeline to classify customer support tickets into predefined categories. This repository translates the dissertation proposal _“Leveraging NLP to Automate Customer Support Workflows”_ into an executable project scaffold with reproducible training scripts, evaluation tooling, and extensible configuration.

## Features
- TF-IDF based preprocessing with optional lemmatisation and stop-word filtering.
- Baseline Logistic Regression and Random Forest classifiers with configurable hyperparameters.
- Grid search support and automated evaluation reports (metrics + confusion matrix).
- Command line utilities for training and running ad-hoc predictions.
- Tailwind CSS powered Flask web app with SQLite logging for interactive model testing.
- Ready-to-extend structure for data experiments, notebooks, and future deep learning upgrades.

## Project Structure
```
.
|-- config/                # YAML configs for data paths, preprocessing, and models
|-- data/
|   |-- raw/               # Houses consumer_complaints.json (see data/raw/README.md)
|   `-- processed/         # Reserved for cleaned features (ignored by git)
|-- models/                # Persisted models (ignored by git)
|-- notebooks/             # Exploratory analysis notebooks
|   `-- evaluation.ipynb   # Runnable online evaluation workflow
|-- scripts/               # Helper shell scripts (e.g. start_app.sh)
|-- src/customer_support_classifier/
|   |-- config.py          # YAML loader helpers
|   |-- data.py            # Dataset loading utilities (CSV + JSON aware)
|   |-- evaluation.py      # Metrics and artefact writers
|   |-- modeling.py        # Model pipeline builders
|   |-- preprocessing.py   # NLP preprocessing toolkit
|   |-- predict.py         # CLI for inference
|   `-- train.py           # CLI for training + grid search
|-- webapp/                # Flask UI for manual ticket classification
`-- requirements.txt       # Python dependencies
```

### Key Files & Directories
- `config/default.yaml` — master configuration controlling data paths, preprocessing flags, and model/grid-search options.
- `src/customer_support_classifier/train.py` — main training CLI; handles train/test split, optional grid search, and artefact export.
- `src/customer_support_classifier/predict.py` — command-line predictor for saved `.joblib` artefacts (single text or batch files).
- `src/customer_support_classifier/preprocessing.py` — reusable text normalisation, stop-word removal, lemmatisation, and TF-IDF vectorisation.
- `src/customer_support_classifier/modeling.py` — builds the scikit-learn pipelines used by both training and inference.
- `webapp/app.py` — Flask factory plus SQLite integrations for storing, displaying, and clearing prediction history.
- `webapp/templates/` — Tailwind CSS templates backing the web UI (form, results panel, history sidebar).
- `scripts/start_app.sh` — convenience launcher that sources `.venv`, sets sane defaults, and starts `flask run`.
- `artifacts/` — output directory for trained models, confusion matrices, and `training_results.json` (auto-populated by training runs).

## Getting Started
1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS / Linux
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare your dataset**
   - The repository now ships with the CFPB consumer complaints JSON export (`data/raw/consumer_complaints.json`).
   - By default the pipeline trains on `_source.complaint_what_happened` (text) and `_source.product` (label).
   - To point at a different dataset or column layout, edit `config/default.yaml` (e.g., update `raw_path`, `format`, `record_key`, `text_column`, `label_column`).
   - The `limit_records` option keeps experiments lightweight (set to `null` to use the full corpus).

## Training & Evaluation
1. **Activate the environment**
   ```bash
   source .venv/bin/activate
   ```
2. **Fast baseline training (no grid search)**
   ```bash
   python -m src.customer_support_classifier.train --config config/default.yaml --no-grid-search
   ```
   - Saves updated `.joblib` models, confusion matrices, and metrics JSON to `artifacts/`.
   - Target a single algorithm with `--model logistic_regression` or `--model random_forest`.
3. **Full grid-search training (longer runtime)**
   ```bash
   python -m src.customer_support_classifier.train --config config/default.yaml
   ```
4. **Review evaluation outputs**
   ```bash
   jupyter notebook notebooks/evaluation.ipynb
   ```
   The notebook reloads artefacts, prints precision/recall/F1 summaries, plots confusion matrices, and runs sample predictions.

## Command-line Inference
- Classify snippets directly:
  ```bash
  python -m src.customer_support_classifier.predict --model-path artifacts/logistic_regression.joblib --text "Customer card declined at POS"
  ```
- Score a file (one ticket per line) and save results:
  ```bash
  python -m src.customer_support_classifier.predict \
      --model-path artifacts/random_forest.joblib \
      --input-file samples/tickets.txt \
      --output tmp/predictions.json
  ```

## Web App
Launch the interactive UI to test models and track predictions:
```bash
export FLASK_APP=webapp.app  # PowerShell: $env:FLASK_APP = "webapp.app"
flask run
```
- Or run `./scripts/start_app.sh` to auto-activate `.venv`, export defaults, and start the server on port 5000 (append `--port 5002` or other `flask run` flags as needed).
- Paste a support ticket description, select a `.joblib` artefact from `artifacts/`, and submit to view predictions plus top probabilities.
- Each prediction is archived in `webapp/predictions.db` (SQLite). Use the “Clear” button in the sidebar to wipe the history.
- Tailwind CSS is served via CDN, so no additional build step is required.

### Quick Launch Script Usage
The helper script wraps the environment activation and `flask run` command:
1. From the project root, ensure the script is executable (only needed once):  
   ```bash
   chmod +x scripts/start_app.sh
   ```
2. Start the development server with sensible defaults:  
   ```bash
   ./scripts/start_app.sh
   ```
   This activates `.venv` (if present), exposes `MODEL_ARTEFACT_DIR=artifacts`, and listens on port `5000`.
3. Pass any additional `flask run` flags—e.g., a different port or host:  
   ```bash
   ./scripts/start_app.sh --port 5002 --host 0.0.0.0
   ```
4. Stop the server with `Ctrl+C` when you're done.

## Evaluation Notebook
Open `notebooks/evaluation.ipynb` to explore the dataset, recreate evaluation metrics, and generate sample predictions in hosted notebook environments (e.g. Colab, Kaggle). Adjust the artefact selection cell if you store models in a custom location.

## Next Steps
- Add exploratory data analysis notebooks under `notebooks/`.
- Remove or adjust the `limit_records` safeguard in `config/default.yaml` once you are ready to train on the entire dataset.
- Extend `grid_search` settings in `config/default.yaml` for deeper optimisation.
- Introduce deep learning baselines (e.g., transformer encoders) once infrastructure permits.
- Integrate evaluation dashboards or REST services to serve predictions downstream.
