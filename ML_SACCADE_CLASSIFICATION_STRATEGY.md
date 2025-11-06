# Machine Learning-Based Saccade Classification Strategy

## Overview

This document outlines a high-level strategy for implementing ML-based saccade classification with a GUI for manual annotation and filtering. The system will classify saccades into **four distinct categories**:
1. **Compensatory**: Saccades in bouts with slow eye movements between them
2. **Purely Orienting**: Freestanding saccades with stable gaze before/after
3. **Saccade-and-Fixate**: First saccades of compensatory bouts (transitional type)
4. **Non-Saccade**: False positives that should be excluded from analysis

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  1. Saccade Detection (existing)                            │
│  2. Feature Extraction (enhanced)                          │
│  3. ML Classification (new)                                 │
│  4. GUI Annotation Tool (new)                                │
│  5. Model Training/Retraining (new)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Enhanced Feature Extraction

### Current Features (Keep & Enhance)
From existing `classify_saccades_orienting_vs_compensatory`:
- `pre_saccade_mean_velocity` (px/s)
- `pre_saccade_position_drift` (px)
- `post_saccade_position_variance` (px²)
- `post_saccade_position_change` (px)
- `bout_size` (count)
- `amplitude` (px)
- `duration` (s)
- `velocity` (peak velocity, px/s)
- `direction` (upward/downward)

### New Features to Add

#### Temporal Context Features
1. **Time since previous saccade** (s)
2. **Time until next saccade** (s)
3. **Position in bout** (1st, 2nd, 3rd, etc. or isolated)
4. **Bout duration** (total time span of bout)
5. **Inter-saccade interval** (mean/std within bout)

#### Velocity Profile Features
6. **Peak velocity** (already exists, but extract separately)
7. **Time to peak velocity** (normalized by duration)
8. **Acceleration phase duration** (time from onset to peak)
9. **Deceleration phase duration** (time from peak to offset)
10. **Velocity asymmetry** (acceleration vs deceleration ratio)
11. **Velocity smoothness** (variance of velocity derivative)

#### Position Profile Features
12. **Position change rate** (amplitude / duration)
13. **Position stability pre-saccade** (variance in pre-window)
14. **Position stability post-saccade** (variance in post-window)
15. **Fixation quality** (if post-saccade variance < threshold)
16. **Drift rate pre-saccade** (drift / pre-window duration)

#### Amplitude & Direction Features
17. **Amplitude consistency in bout** (std of amplitudes in bout)
18. **Direction alternation in bout** (alternating vs same direction)
19. **Amplitude relative to bout mean** (normalized)
20. **Direction relative to previous** (same/opposite)

#### Contextual Features
21. **Is first in bout** (boolean)
22. **Is last in bout** (boolean)
23. **Is isolated** (boolean)
24. **Bout density** (saccades per second in bout)
25. **Classification confidence** (from rule-based, if available)

### Feature Engineering Considerations
- **Normalization**: Normalize features by amplitude or recording statistics where appropriate
- **Relative features**: Use ratios/percentages for amplitude-independent features
- **Temporal features**: Use time-based features (seconds) rather than frame-based for FPS independence
- **Missing data**: Handle edge cases (first/last saccades) with appropriate defaults or flags

---

## Phase 2: Machine Learning Model Design

### Model Architecture Options

#### Option A: Random Forest Classifier (Recommended for Start)
**Pros:**
- Handles non-linear relationships well
- Provides feature importance
- Robust to outliers
- No feature scaling required
- Interpretable

**Cons:**
- May not capture complex temporal patterns
- Limited to tabular features

**Implementation:**
- Use `sklearn.ensemble.RandomForestClassifier`
- Start with 100-200 trees, max_depth=10-15
- Use class_weight='balanced' for imbalanced data

#### Option B: Gradient Boosting (XGBoost/LightGBM)
**Pros:**
- Often better performance than RF
- Handles imbalanced data well
- Feature importance available

**Cons:**
- More hyperparameter tuning needed
- Less interpretable than RF

#### Option C: Neural Network (SELECTED)
**Pros:**
- Can learn complex patterns
- Can incorporate time series directly
- Works well with M2 Mac (Metal acceleration) and Ubuntu (CUDA/CPU)
- Can handle non-linear feature interactions

**Cons:**
- Requires more data
- Less interpretable
- More complex to implement

**Implementation:**
- Use TensorFlow/Keras or PyTorch (TensorFlow recommended for M2 Mac compatibility)
- Multi-layer perceptron (MLP) for tabular features
- Optional: LSTM/GRU layer for time series segments
- Use Metal Performance Shaders (MPS) backend on M2 Mac
- Use CPU or CUDA on Ubuntu
- Start with 2-3 hidden layers, 64-128 units per layer
- Dropout for regularization (0.2-0.3)
- Batch normalization
- Class weights for imbalanced data

### Classification Strategy

#### Four-Class Problem
1. **Compensatory**: In bouts, with drift/movement between saccades
2. **Purely Orienting**: Isolated, stable before/after
3. **Saccade-and-Fixate**: First saccade of compensatory bout (or isolated with fixate pattern)
4. **Non-Saccade**: False positive detections (blinks, artifacts, noise)

#### Hierarchical Approach (Alternative)
1. **First level**: Compensatory vs Non-Compensatory (binary)
2. **Second level**: For Non-Compensatory → Orienting vs Saccade-and-Fixate

**Rationale**: Saccade-and-Fixate may be easier to distinguish from Orienting than from Compensatory.

### Model Training Pipeline

```
1. Load labeled data (from GUI annotations)
2. Feature extraction for all saccades
3. Train/test split (80/20 or 70/30)
4. Feature selection (optional, based on importance)
5. Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
6. Cross-validation (5-fold)
7. Model evaluation (confusion matrix, precision/recall per class)
8. Save model (joblib/pickle)
```

---

## Phase 3: GUI Annotation Tool

### GUI Requirements

#### Main Window Layout
```
┌─────────────────────────────────────────────────────────────┐
│  Saccade Classification Annotation Tool                      │
├─────────────────────────────────────────────────────────────┤
│  [File Menu] [View Menu] [Tools Menu]                       │
├──────────────┬──────────────────────────────────────────────┤
│              │                                               │
│  Saccade     │  Time Series Plot                            │
│  List        │  (Position + Velocity)                       │
│              │                                               │
│  [Filters]   │  [Previous] [Next] [Play]                    │
│              │                                               │
│  Selected:   │  Peri-Saccade Segment Plot                   │
│  #42         │  (Position + Velocity aligned)               │
│              │                                               │
│  [Classify]  │  Feature Values Display                      │
│  [Exclude]   │  [Save Annotation]                           │
│              │                                               │
└──────────────┴───────────────────────────────────────────────┘
```

### Key Features

#### 1. Saccade List Panel
- **Display**: Table/list of all detected saccades
- **Columns**: ID, Time, Amplitude, Duration, Current Label, Confidence
- **Filtering**: By class, confidence, amplitude range, time range
- **Sorting**: By time, amplitude, confidence, etc.
- **Selection**: Click to select, highlight in plots

#### 2. Time Series Plot
- **Display**: Full recording with position and velocity traces
- **Overlay**: 
  - Vertical lines/rectangles for detected saccades
  - Color-coded by current classification
  - Highlight selected saccade
- **Navigation**: Zoom, pan, time range selection
- **Interaction**: Click on saccade to select

#### 3. Peri-Saccade Segment Plot
- **Display**: Position and velocity traces aligned to threshold crossing
- **Window**: Pre-saccade and post-saccade periods
- **Features**: Show baseline window, saccade duration
- **Comparison**: Option to overlay similar saccades

#### 4. Classification Controls
- **Class Buttons**: 
  - [Compensatory] [Purely Orienting] [Saccade-and-Fixate] [Non-Saccade] [Unclassified]
- **Note**: "Non-Saccade" replaces the separate "Exclude" button - it's now a classification class
- **Confidence Slider**: Manual confidence adjustment (optional)
- **Keyboard Shortcuts**: 
  - `1` = Compensatory, `2` = Orienting, `3` = Saccade-and-Fixate, `4` = Non-Saccade
  - `N` = Next, `P` = Previous, `S` = Save

#### 5. Feature Display Panel
- **Show**: All extracted features for selected saccade
- **Format**: Table or formatted text
- **Highlight**: Features that strongly indicate each class

#### 6. Batch Operations
- **Auto-classify**: Run ML model on unlabeled saccades
- **Filter by confidence**: Show only low-confidence predictions
- **Export/Import**: Save/load annotations (CSV/JSON)
- **Statistics**: Show class distribution, annotation progress

### GUI Technology Stack

#### Option A: PyQt5/PyQt6 (Recommended)
**Pros:**
- Native look and feel
- Good plotting integration (matplotlib)
- Mature and stable
- Cross-platform

**Cons:**
- Steeper learning curve
- More verbose code

#### Option B: Tkinter
**Pros:**
- Built into Python
- Simple to use
- Lightweight

**Cons:**
- Less modern UI
- Limited plotting capabilities

#### Option C: Web-based (Flask/Dash + Plotly)
**Pros:**
- Modern UI
- Easy to deploy/share
- Interactive plots (Plotly)

**Cons:**
- Requires web server
- More complex architecture

**Recommendation**: Start with PyQt5 for desktop app, consider web-based later for sharing.

### Data Storage

#### Annotation File Format (CSV)
```csv
experiment_id,saccade_id,time,amplitude,duration,user_label,user_confidence,notes,annotation_date
exp_001,1,12.345,45.2,0.125,compensatory,0.9,"First in bout",2024-01-15
exp_001,2,12.567,38.1,0.098,compensatory,0.85,"",2024-01-15
exp_002,1,15.234,52.3,0.142,orienting,0.95,"Isolated",2024-01-16
exp_002,5,18.456,12.3,0.045,non_saccade,0.8,"Artifact",2024-01-16
```

**Key Points:**
- `experiment_id`: Identifies which notebook run/experiment this saccade came from
- `user_label`: One of: `compensatory`, `orienting`, `saccade_and_fixate`, `non_saccade`
- All annotations are stored in a single master file for incremental training

#### Model File Format
- **Model**: TensorFlow SavedModel or HDF5 format (`.h5`)
- **Metadata**: JSON file with:
  - Feature list and order
  - Model architecture (layers, units, activation functions)
  - Training date and version
  - Training statistics (accuracy, loss, per-class metrics)
  - Number of training samples per class
  - Feature scaler parameters (StandardScaler/MinMaxScaler)
- **Feature scaler**: Saved as part of model or separate pickle file
- **Versioning**: Models saved as `saccade_classifier_v1.h5`, `saccade_classifier_v2.h5`, etc.

---

## Phase 4: Workflow Integration

### Integration with Existing Pipeline

#### Modified Notebook Flow
```python
# 1. Detect saccades (existing)
saccade_results = analyze_eye_video_saccades(...)

# 2. Extract enhanced features (new function)
enhanced_features = extract_ml_features(
    saccade_results, 
    df, 
    fps
)

# 3. Run rule-based classification first (for initial labels)
saccade_results = classify_with_rule_based(saccade_results, ...)  # Existing function

# 4. Load or train ML model
if model_exists:
    model = load_ml_model('saccade_classifier_v1.h5')
    predictions = model.predict(enhanced_features)
    # Use rule-based predictions as fallback for low-confidence ML predictions
    low_conf_mask = model.predict_proba(enhanced_features).max(axis=1) < 0.5
    predictions[low_conf_mask] = saccade_results['rule_based_class'][low_conf_mask]
else:
    print("No ML model found. Using rule-based classification.")
    print("Use GUI to annotate data and train ML model.")
    predictions = saccade_results['rule_based_class']  # Use rule-based as initial labels

# 4. Add predictions to dataframe
all_saccades_df['ml_class'] = predictions
all_saccades_df['ml_confidence'] = model.predict_proba(enhanced_features).max(axis=1)

# 5. Optionally open GUI for review/editing
if review_in_gui:
    launch_annotation_gui(saccade_results, enhanced_features, model)
```

### New Functions to Add

#### `extract_ml_features(saccade_results, df, fps)`
- Extract all features (existing + new)
- Return DataFrame with one row per saccade
- Handle edge cases (first/last saccades)

#### `train_ml_classifier(annotations_file_path, features_df, model_version=None)`
- Load ALL annotations from master annotations file (across all experiments)
- Merge with features from current experiment
- Handle incremental training: append new annotations to existing dataset
- Train neural network model on combined dataset
- Evaluate with cross-validation
- Save model with version number
- Return training statistics

**Incremental Training Strategy:**
1. Load master annotations file (contains all labeled saccades from all experiments)
2. Load features for all experiments (or extract on-the-fly)
3. Combine all labeled data
4. Retrain model from scratch on full dataset (better than fine-tuning for this use case)
5. Save new model version
6. Keep old model versions for comparison

#### `classify_with_ml(features_df, model)`
- Load model
- Predict classes and confidence
- Return predictions

#### `launch_annotation_gui(saccade_results, features_df, experiment_id, model=None, annotations_file=None)`
- Initialize GUI
- Load saccade data for current experiment
- Pre-populate with rule-based classifications (if no ML model) or ML predictions
- Display plots and controls
- Allow user to review and correct classifications
- Save annotations to master annotations file (append mode)
- Track `experiment_id` to identify source of each annotation

---

## Phase 5: Implementation Plan

### Step 1: Enhanced Feature Extraction (Week 1)
- [ ] Implement `extract_ml_features()` function
- [ ] Add new temporal context features
- [ ] Add velocity profile features
- [ ] Add position profile features
- [ ] Test feature extraction on sample data
- [ ] Document feature definitions

### Step 2: Neural Network ML Pipeline (Week 2)
- [ ] Implement `train_ml_classifier()` function with TensorFlow/Keras
- [ ] Design MLP architecture (2-3 hidden layers, dropout, batch norm)
- [ ] Implement feature scaling (StandardScaler)
- [ ] Implement train/test split and cross-validation
- [ ] Add model evaluation metrics (confusion matrix, per-class F1)
- [ ] Save/load model functionality (HDF5 format)
- [ ] Implement incremental training (load all annotations, retrain)
- [ ] Test on synthetic or small real dataset
- [ ] Ensure M2 Mac (MPS) and Ubuntu (CPU) compatibility

### Step 3: GUI Development (Weeks 3-4)
- [ ] Design GUI layout (mockup)
- [ ] Implement main window with plots
- [ ] Add saccade list panel
- [ ] Add classification controls
- [ ] Implement navigation (next/previous)
- [ ] Add annotation saving/loading
- [ ] Test GUI with sample data

### Step 4: Integration (Week 5)
- [ ] Integrate ML classification into notebook
- [ ] Add GUI launch option
- [ ] Test end-to-end workflow
- [ ] Handle edge cases and errors
- [ ] Add documentation

### Step 5: Refinement (Week 6+)
- [ ] Collect initial annotations
- [ ] Train first model
- [ ] Iterate on features based on model importance
- [ ] Improve GUI based on user feedback
- [ ] Add advanced features (batch operations, filtering)

---

## Key Design Decisions

### 1. Three-Class vs Hierarchical Classification
**Decision**: Start with three-class, evaluate performance. Can switch to hierarchical if needed.

**Rationale**: Simpler to implement and interpret initially. Can optimize later.

### 2. Feature Normalization
**Decision**: Normalize amplitude-dependent features by amplitude. Keep time-based features in seconds.

**Rationale**: Makes model more robust to different recording conditions.

### 3. Handling Edge Cases
**Decision**: 
- First saccade: Use available pre-window data, flag if incomplete
- Last saccade: Use available post-window data, flag if incomplete
- Very short bouts: Treat as isolated for some features

**Rationale**: Don't exclude edge cases, but flag them for potential manual review.

### 4. Model Update Strategy
**Decision**: 
- Initial: Train on manually annotated subset
- Iterative: Retrain as more annotations accumulate
- Version control: Save model versions with metadata

**Rationale**: Model improves with more data, but need to track changes.

### 5. GUI Annotation Workflow
**Decision**: 
- Start with rule-based predictions pre-loaded (from existing classification)
- User reviews and corrects
- Save all annotations to master file (with experiment_id)
- Show confidence to prioritize review
- After sufficient annotations, train ML model
- Future runs: Start with ML predictions, fallback to rule-based for low confidence

**Rationale**: Rule-based provides good starting point. ML improves as more data is annotated.

---

## Success Metrics

### Model Performance
- **Accuracy**: >85% on test set
- **Per-class F1**: >0.8 for each class (including Non-Saccade)
- **Confusion matrix**: Low confusion between classes
- **Non-Saccade recall**: >0.9 (important to catch false positives)

### Annotation Efficiency
- **Time per saccade**: <30 seconds average
- **Annotation coverage**: >80% of saccades labeled
- **Inter-annotator agreement**: >90% (if multiple annotators)

### User Experience
- **GUI responsiveness**: <100ms for navigation
- **Plot clarity**: Clear visualization of saccade features
- **Workflow smoothness**: Minimal clicks to classify

---

## Future Enhancements

1. **Active Learning**: Model suggests which saccades to annotate next
2. **Transfer Learning**: Pre-train on one dataset, fine-tune on another
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Time Series Features**: Use LSTM/GRU for temporal patterns
5. **Multi-eye Analysis**: Classify using both eyes simultaneously
6. **Real-time Classification**: Classify during recording

---

## Questions for Discussion

1. **Class Definitions**: Are the three classes well-defined? Should "Saccade-and-Fixate" be a separate class or a subtype?

2. **Annotation Strategy**: Should we start with rule-based predictions and correct, or annotate from scratch?

3. **Feature Priority**: Which new features are most important to add first?

4. **GUI Complexity**: Should we start simple and add features, or build comprehensive GUI from start?

5. **Model Selection**: Random Forest vs XGBoost vs Neural Network for initial implementation?

6. **Data Requirements**: How many annotated saccades do we need for reliable training? (Estimate: 100-200 per class minimum)

7. **Integration Point**: Should ML classification replace rule-based, or run in parallel for comparison?

**RESOLVED**: Start with rule-based for initial annotation. ML replaces rule-based once trained, but rule-based used as fallback for low-confidence ML predictions.

