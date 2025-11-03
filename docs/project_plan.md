# Project Plan

## Research Objective
Evaluate whether Natural Language Processing techniques can automatically route customer support tickets to predefined categories with accuracy and efficiency comparable to, or better than, manual triage.

## Scope
- Dataset of historical tickets with textual descriptions and categorical labels.
- Baseline models: Logistic Regression and Random Forest using TF-IDF features.
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix analysis, and qualitative error inspection.

## Methodology Outline
1. **Data Preparation**
   - Clean text (normalise case, remove punctuation, optionally remove stop words).
   - Tokenise and lemmatise text.
   - Convert to TF-IDF features.
2. **Model Training**
   - Split dataset into train/test using stratified sampling.
   - Train baseline classifiers.
   - Run grid search to optimise hyperparameters.
3. **Evaluation & Interpretation**
   - Generate classification metrics and confusion matrices.
   - Analyse misclassifications to identify ambiguous or multi-intent tickets.
   - Document findings for dissertation chapters.
4. **Operationalisation (Stretch)**
   - Package trained model for batch scoring or API deployment.
   - Explore adding sentiment or priority scoring.

## Deliverables
- Reproducible training pipeline (`train.py`).
- Prediction interface (`predict.py`).
- Configurable experiment settings (`config/default.yaml`).
- Artefacts directory containing models, metrics, and plots per run.
- Documentation for set-up, usage, and future extensions.

## Timeline (Indicative)
- **Week 1-2:** Data acquisition, cleansing, exploratory analysis.
- **Week 3-4:** Implement baseline models, run initial evaluations.
- **Week 5:** Hyperparameter tuning, detailed error analysis, iterate on features.
- **Week 6:** Compile findings, document outcomes, plan extensions (e.g., deep learning).
