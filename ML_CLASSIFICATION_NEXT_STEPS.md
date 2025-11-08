# ML Saccade Classification: Next Steps

## Current Context
- The working notebook `SANDBOX_1_1_Loading_+_Saccade_detection.py` builds `saccade_results`, a per-eye dictionary holding smoothed position/velocity series, detected saccade summaries (direction-specific DataFrames), peri-saccade segments, and metadata needed downstream.
- Feature extraction produces per-saccade tables (`features_v1`, `features_v2`) which are concatenated into `features_combined` before launching the GUI. The GUI call passes the full `saccade_results` and combined feature set, so the interface currently preloads every peri-saccade segment.
- The refactoring goal in `REFACTORING_PLAN_ML_CLASSIFICATION.md` is to migrate classification to a scikit-learn Random Forest pipeline while preserving label encoding helpers and adding feature-importance tooling.

## Operational Goals
- With ~100 experiments to process, the aim is to train the classifier only a handful of times, then run batch inference silently across experiments.
- When misclassifications are discovered, the workflow should allow appending new annotations and retraining the model incrementally without rebuilding from scratch.

## Observations
- GUI startup lag is likely caused by constructing the combined DataFrame of all saccades and passing the full dict of peri-saccade segments into `launch_annotation_gui`. Each segment currently includes multi-column time series, so thousands of segments lead to large in-memory structures that the GUI must sort and render before becoming interactive.
- The notebook already saves intermediate CSV/HTML artifacts (QC figures, blink summaries, interpolated coordinates), giving us reproducible inputs for a dedicated `ml_classification.py` module.
- Feature categories are well defined via `visualize_ml_features`, and rule-based classifications plus GUI labels can serve as training targets for the new Random Forest model.

## Proposed Action Plan
1. **Data Packaging**
   - Isolate a minimal schema (e.g., `saccade_id`, `eye`, `perisaccade_trace`, key scalars) for GUI consumption to reduce warm-up cost.
   - Create a serializable dataset export (Parquet/Feather) from `features_combined` plus human annotations for model training.
2. **Model Refactor Implementation**
   - Replace TensorFlow dependencies with scikit-learn Random Forest per the refactoring plan, keeping `encode_labels`, `decode_labels`, and `compute_class_weights` utilities intact.
   - Implement the new helper functions (`build_random_forest_model`, feature-importance utilities) with rich docstrings and unit-style smoke tests.
3. **Training & Validation Workflow**
   - Define a reproducible training script/notebook that loads exported features, splits train/validation folds, and saves model artifacts (joblib).
   - Integrate performance diagnostics (confusion matrix, per-class precision/recall) and tie them back into the GUI for quick review of misclassifications.
   - Support periodic retraining that ingests additive annotation batches so large-scale silent runs only require occasional updates.
4. **GUI & Pipeline Integration**
   - Add lazy-loading or on-demand fetching of peri-saccade segments in the GUI to avoid caching the full set on launch.
   - Surface Random Forest predictions and feature-importance summaries within the GUI to guide manual reviewers.
   - Provide a non-interactive batch mode that consumes saved model artifacts and processes multiple experiments unattended.
5. **Documentation & Automation**
   - Document the end-to-end pipeline (data extraction → feature export → model training → GUI-assisted review → batch inference) in project docs.
   - Prepare unit/integration tests to cover label encoding, model training with sample data, and regression tests for the refactored module.
