# ML-Based Saccade Classification - Implementation Plan

## Current Status

### ✅ Completed
1. **Feature Extraction** (`sleap/ml_feature_extraction.py`)
   - ~36 features implemented across Categories A, B, C, D, G, H
   - Experiment ID generation from data_path
   - Edge case handling (first/last saccades, incomplete windows)

2. **Feature Visualization** (`sleap/visualization.py`)
   - Violin plots by category
   - Class comparison plots
   - Normalization and diagnostic outputs

3. **Rule-Based Classification** (existing)
   - 2 classes: orienting, compensatory
   - Provides initial labels for ML training

### ⏳ Next Steps

---

## Implementation Plan

### Phase 1: Neural Network Model Architecture (Priority: High)

**File**: `sleap/ml_classification.py` (NEW)

**Functions to implement:**

1. **`build_mlp_model(n_features, n_classes=4, hidden_units=[128, 64, 32], dropout_rate=0.3)`**
   - Build TensorFlow/Keras MLP model
   - Handle M2 Mac (MPS) and Ubuntu (CPU/CUDA)
   - Return compiled model

2. **`setup_tensorflow_backend()`**
   - Detect available backend (MPS/GPU/CPU)
   - Configure TensorFlow accordingly
   - Return backend info

3. **`compute_class_weights(y_one_hot)`**
   - Calculate balanced class weights for imbalanced data
   - Return dictionary for TensorFlow

**Architecture:**
```
Input Layer: (n_features,)
├── Dense(128) + ReLU + BatchNorm + Dropout(0.3)
├── Dense(64) + ReLU + BatchNorm + Dropout(0.2)
├── Dense(32) + ReLU + Dropout(0.2) [optional]
└── Dense(4) + Softmax
```

**Key Requirements:**
- M2 Mac: Use MPS backend (Metal Performance Shaders)
- Ubuntu: Use CPU or CUDA if available
- Cross-platform compatibility
- Early stopping callback
- Model checkpointing

---

### Phase 2: Training Pipeline (Priority: High)

**Functions to implement:**

1. **`prepare_training_data(annotations_df, features_dict)`**
   - Load annotations from master CSV
   - Merge with features by `experiment_id` + `saccade_id`
   - Handle missing features (extract on-the-fly if needed)
   - Return X (features), y (labels), feature_names

2. **`train_ml_classifier(annotations_file_path, features_dict, model_version=None, output_dir='sleap/models')`**
   - Load ALL annotations (incremental training)
   - Prepare training data
   - Split train/test (80/20)
   - Scale features (StandardScaler)
   - Build model
   - Train with validation
   - Evaluate metrics (accuracy, per-class F1, confusion matrix)
   - Save model + scaler + metadata
   - Return model, training_stats

3. **`evaluate_model(model, X_test, y_test, class_names)`**
   - Compute accuracy, per-class precision/recall/F1
   - Generate confusion matrix
   - Return metrics dictionary

**Incremental Training Strategy:**
- Load master annotations file (all experiments)
- Extract/load features for all annotated saccades
- Retrain from scratch on full dataset
- Save new model version (v1, v2, v3, ...)
- Keep old versions for comparison

---

### Phase 3: Model Saving/Loading (Priority: High)

**Functions to implement:**

1. **`save_model(model, scaler, feature_names, training_stats, model_version, output_dir)`**
   - Save model as HDF5 (`.h5`)
   - Save scaler as pickle (`.pkl`)
   - Save metadata as JSON:
     ```json
     {
       "model_version": 1,
       "training_date": "2025-01-15T10:30:00",
       "n_features": 36,
       "feature_names": [...],
       "n_samples": 150,
       "n_samples_per_class": {"compensatory": 50, "orienting": 60, ...},
       "training_stats": {
         "train_accuracy": 0.92,
         "test_accuracy": 0.88,
         "per_class_f1": {...}
       },
       "model_architecture": {...}
     }
     ```

2. **`load_model(model_path)`**
   - Load model from HDF5
   - Load scaler from pickle
   - Load metadata from JSON
   - Return model, scaler, metadata

3. **`get_latest_model_version(output_dir)`**
   - Find highest version number
   - Return version number or None

---

### Phase 4: Prediction Function (Priority: High)

**Functions to implement:**

1. **`predict_saccades(model, scaler, features_df, confidence_threshold=0.5)`**
   - Scale features using saved scaler
   - Predict classes and probabilities
   - Return predictions DataFrame with columns:
     - `ml_class`: Predicted class (0-3)
     - `ml_class_label`: Human-readable label
     - `ml_confidence`: Max probability
     - `ml_probabilities`: All class probabilities

2. **`classify_with_ml_fallback(features_df, model, scaler, rule_based_predictions, confidence_threshold=0.5)`**
   - Get ML predictions
   - Use rule-based as fallback for low-confidence predictions
   - Return combined predictions

---

### Phase 5: Annotation Storage Infrastructure (Priority: Medium)

**File**: `sleap/annotation_storage.py` (NEW)

**Functions to implement:**

1. **`initialize_annotations_file(annotations_file_path)`**
   - Create master CSV file with headers if doesn't exist
   - Columns: `experiment_id`, `saccade_id`, `time`, `amplitude`, `duration`, 
              `user_label`, `user_confidence`, `notes`, `annotation_date`, `annotator`

2. **`save_annotation(annotations_file_path, experiment_id, saccade_id, user_label, user_confidence=1.0, notes='')`**
   - Append annotation to master file
   - Check for duplicates (experiment_id + saccade_id)
   - Update if exists, append if new

3. **`load_annotations(annotations_file_path, experiment_id=None)`**
   - Load all annotations or filter by experiment_id
   - Return DataFrame

4. **`get_annotation_stats(annotations_file_path)`**
   - Count annotations per class
   - Count annotations per experiment
   - Return statistics dictionary

---

### Phase 6: GUI Annotation Tool (Priority: Medium)

**File**: `sleap/annotation_gui.py` (NEW)

**Technology**: PyQt5

**Dual Purpose:**
1. **Generate Training Data**: Start with rule-based classifications, allow user to correct/annotate
2. **Verify ML Classification**: After ML model is trained, display ML predictions for user verification

**Key Components:**

1. **Main Window Class: `SaccadeAnnotationGUI`**
   - Two-panel layout (list + plots)
   - Saccade list widget (QTableWidget or QListWidget)
   - Matplotlib plots embedded (QWidget)
   - Classification buttons
   - Navigation controls
   - Mode indicator (Training Mode / Verification Mode)

2. **Plot Widgets:**
   - Time series plot (position + velocity with saccades highlighted)
   - Peri-saccade segment plot (aligned position/velocity)
   - Color coding: Current label, rule-based prediction, ML prediction (if available)

3. **Features:**

   **Training Mode (Initial):**
   - Pre-populate with rule-based classifications
   - Display rule-based confidence
   - User corrects classifications
   - Save annotations to master file
   - Show feature values for selected saccade
   - Filter by rule-based class, confidence, amplitude
   - Statistics: "X saccades annotated, Y remaining"

   **Verification Mode (After ML Training):**
   - Display ML predictions (primary)
   - Display rule-based predictions (for comparison)
   - Highlight disagreements between ML and rule-based
   - User verifies/corrects ML predictions
   - Show ML confidence scores
   - Filter by ML class, ML confidence, disagreements
   - Statistics: "ML accuracy: X%, disagreements: Y"

   **Common Features:**
   - Keyboard shortcuts:
     - `1` = Compensatory
     - `2` = Orienting (Purely Orienting)
     - `3` = Saccade-and-Fixate
     - `4` = Non-Saccade
     - `N` = Next saccade
     - `P` = Previous saccade
     - `S` = Save annotation
     - `F` = Filter dialog
   - Save annotations to master file
   - Export/import annotations
   - Progress tracking

**GUI Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Saccade Classification Tool - [Training Mode]             │
├──────────────┬──────────────────────────────────────────────┤
│              │  Time Series Plot                            │
│  Saccade     │  [Position + Velocity traces]               │
│  List        │  [Saccades highlighted by current label]    │
│  (scrollable)│                                              │
│              │  Peri-Saccade Segment                        │
│  Filters:    │  [Aligned position/velocity]                │
│  [Class ▼]   │                                              │
│  [Confidence]│  Current Label: [Compensatory]              │
│  [Amplitude] │  Rule-Based: [Compensatory] (conf: 0.85)    │
│              │  ML Prediction: [Orienting] (conf: 0.72)    │
│  Statistics: │                                              │
│  45/120 done │  Classification:                           │
│              │  [1]Compensatory [2]Orienting               │
│              │  [3]Saccade-Fixate [4]Non-Saccade            │
│              │  [Save] [Next] [Previous]                   │
└──────────────┴──────────────────────────────────────────────┘
```

**Minimal Viable GUI:**
- Simple, functional interface
- Focus on annotation efficiency
- Clear display of rule-based vs ML predictions
- Easy navigation and saving
- Can be enhanced later with advanced features

---

### Phase 7: Integration with Notebook (Priority: Medium)

**Modifications to notebook:**

1. **After saccade detection:**
   ```python
   # Extract features
   features_df = extract_ml_features(...)
   
   # Try to load ML model
   model, scaler, metadata = load_model('sleap/models/saccade_classifier_v1.h5')
   
   if model is not None:
       # Get ML predictions
       predictions = predict_saccades(model, scaler, features_df)
       # Use rule-based as fallback for low confidence
       predictions = classify_with_ml_fallback(...)
   else:
       # Use rule-based only
       predictions = rule_based_predictions
   ```

2. **Add GUI launch option:**
   ```python
   # Optionally launch GUI for review
   if launch_gui:
       from sleap.annotation_gui import launch_annotation_gui
       
       # Training mode: Generate training data from rule-based classifications
       if model is None:
           launch_annotation_gui(
               saccade_results=saccade_results,
               features_df=features_df,
               experiment_id=experiment_id,
               model=None,
               annotations_file='data/annotations/saccade_annotations_master.csv',
               mode='training'
           )
       # Verification mode: Verify ML predictions
       else:
           launch_annotation_gui(
               saccade_results=saccade_results,
               features_df=features_df,
               experiment_id=experiment_id,
               model=model,
               scaler=scaler,
               annotations_file='data/annotations/saccade_annotations_master.csv',
               mode='verification'
           )
   ```

3. **Add training function call:**
   ```python
   # After collecting annotations, train model
   if train_model:
       from sleap.ml_classification import train_ml_classifier
       model, stats = train_ml_classifier(
           annotations_file_path='data/annotations/saccade_annotations_master.csv',
           features_dict={experiment_id: features_df},
           model_version=None  # Auto-increment
       )
   ```

---

## Implementation Order

### Step 1: Core ML Infrastructure (Week 1)
1. ✅ Create `sleap/ml_classification.py`
2. ✅ Implement `setup_tensorflow_backend()` - M2 Mac + Ubuntu support
3. ✅ Implement `build_mlp_model()` - Neural network architecture
4. ✅ Implement `compute_class_weights()` - Handle imbalanced data
5. ✅ Test model building and compilation

### Step 2: Training Pipeline (Week 1-2)
1. ✅ Implement `prepare_training_data()` - Load and merge annotations + features
2. ✅ Implement `train_ml_classifier()` - Full training pipeline
3. ✅ Implement `evaluate_model()` - Metrics computation
4. ✅ Test with synthetic data or small real dataset

### Step 3: Model Persistence (Week 2)
1. ✅ Implement `save_model()` - Save model, scaler, metadata
2. ✅ Implement `load_model()` - Load everything back
3. ✅ Implement `get_latest_model_version()` - Version management
4. ✅ Test save/load cycle

### Step 4: Prediction Functions (Week 2)
1. ✅ Implement `predict_saccades()` - Classify new saccades
2. ✅ Implement `classify_with_ml_fallback()` - ML + rule-based hybrid
3. ✅ Test predictions on sample data

### Step 5: Annotation Storage (Week 2-3)
1. ✅ Create `sleap/annotation_storage.py`
2. ✅ Implement annotation file management functions
3. ✅ Test annotation saving/loading

### Step 6: GUI Development (Week 3-4)
1. ✅ Create `sleap/annotation_gui.py`
2. ✅ Implement basic GUI layout
3. ✅ Add plot widgets (time series + peri-saccade)
4. ✅ Add classification controls
5. ✅ Implement Training Mode:
   - Pre-populate with rule-based classifications
   - Display rule-based confidence
   - Allow user corrections
6. ✅ Implement Verification Mode:
   - Display ML predictions (primary)
   - Display rule-based predictions (comparison)
   - Highlight disagreements
   - Show ML confidence scores
7. ✅ Add navigation and saving
8. ✅ Add statistics display (progress, accuracy)
9. ✅ Test GUI with sample data (both modes)

### Step 7: Integration (Week 4-5)
1. ✅ Add ML classification to notebook workflow
2. ✅ Add GUI launch option (both training and verification modes)
3. ✅ Add training function call
4. ✅ Test end-to-end workflow:
   - Training data generation workflow
   - ML training workflow
   - ML verification workflow
5. ✅ Handle edge cases and errors

### Step 8: Testing & Refinement (Week 5+)
1. ✅ Test with real data
2. ✅ Use GUI in Training Mode to collect initial annotations (100-200 saccades)
   - Start with rule-based classifications
   - Correct misclassifications
   - Focus on edge cases and ambiguous saccades
3. ✅ Train first model
4. ✅ Use GUI in Verification Mode to verify ML predictions
   - Review ML predictions
   - Correct errors
   - Collect additional training data from corrections
5. ✅ Retrain model with expanded dataset
6. ✅ Iterate based on performance
7. ✅ Refine GUI based on user feedback

---

## File Structure

```
sleap/
├── ml_classification.py          # NEW: ML model, training, prediction
├── annotation_storage.py          # NEW: Annotation file management
├── annotation_gui.py              # NEW: GUI for manual annotation
├── ml_feature_extraction.py       # EXISTING: Feature extraction
├── visualization.py               # EXISTING: Feature visualization
└── models/                        # NEW: Saved models directory
    ├── saccade_classifier_v1.h5
    ├── scaler_v1.pkl
    ├── model_metadata_v1.json
    └── ...

data/
└── annotations/                  # NEW: Master annotations file
    └── saccade_annotations_master.csv
```

---

## Key Design Decisions

### 1. Model Architecture
- **Choice**: Multi-Layer Perceptron (MLP)
- **Rationale**: Works well with tabular features, simpler than LSTM/CNN
- **Future**: Can add LSTM layer for time series if needed

### 2. Incremental Training
- **Choice**: Retrain from scratch on full dataset
- **Rationale**: Simpler than fine-tuning, avoids catastrophic forgetting
- **Alternative**: Fine-tuning possible but more complex

### 3. Feature Scaling
- **Choice**: StandardScaler (z-score normalization)
- **Rationale**: Already used in visualization, works well with neural networks
- **Note**: Must save scaler with model for prediction

### 4. Class Encoding
- **0**: Compensatory
- **1**: Orienting (Purely Orienting)
- **2**: Saccade-and-Fixate
- **3**: Non-Saccade
- **-1**: Unclassified (for rule-based when classification not performed)

### 5. Confidence Threshold
- **Default**: 0.5 (50%)
- **Usage**: ML predictions below threshold fall back to rule-based
- **Rationale**: Conservative approach, improves reliability

### 6. GUI Technology
- **Choice**: PyQt5
- **Rationale**: Mature, cross-platform, good matplotlib integration
- **Alternative**: Could use PyQt6 or web-based (Dash) later

---

## Testing Strategy

### Unit Tests
- Model building and compilation
- Feature scaling and preparation
- Model save/load
- Prediction functions

### Integration Tests
- End-to-end training pipeline
- Annotation saving/loading
- GUI functionality
- Notebook integration

### Data Tests
- Test with small real dataset (10-20 saccades)
- Test with synthetic data
- Test edge cases (all one class, missing features, etc.)

---

## Success Criteria

### Model Performance
- **Accuracy**: >85% on test set
- **Per-class F1**: >0.8 for each class
- **Non-Saccade recall**: >0.9 (critical for false positive detection)

### Code Quality
- **Modularity**: Clear separation of concerns
- **Documentation**: Docstrings for all functions
- **Error handling**: Graceful handling of edge cases
- **Cross-platform**: Works on M2 Mac and Ubuntu

### User Experience
- **GUI**: Responsive, intuitive interface
- **Workflow**: Smooth annotation process
- **Integration**: Seamless notebook integration

---

## Dependencies

### Required Packages
- `tensorflow` (>=2.10.0 for MPS support on M2 Mac)
- `scikit-learn` (for StandardScaler, train_test_split)
- `pandas` (for data handling)
- `numpy` (for numerical operations)
- `PyQt5` (for GUI)
- `matplotlib` (for plotting in GUI)

### Optional Packages
- `tensorflow-metal` (for M2 Mac MPS support - may be needed)

---

## Next Immediate Steps

1. **Create `sleap/ml_classification.py`** with:
   - TensorFlow backend setup
   - Model building function
   - Basic training function (skeleton)

2. **Test TensorFlow setup** on M2 Mac:
   - Verify MPS backend works
   - Test model compilation
   - Test simple training

3. **Create annotation storage infrastructure**:
   - Master CSV file format
   - Save/load functions

4. **Start with minimal training pipeline**:
   - Load annotations
   - Prepare data
   - Train simple model
   - Save model

5. **Test with synthetic data**:
   - Create small synthetic dataset
   - Train model
   - Verify predictions work

---

## GUI Workflow Details

### Training Mode Workflow (Generate Training Data)

1. **Initial State:**
   - Load saccade results from current experiment
   - Extract features
   - Get rule-based classifications (from existing `classify_saccades_orienting_vs_compensatory`)
   - Display all saccades with rule-based labels pre-populated

2. **User Actions:**
   - Navigate through saccades (Next/Previous buttons or keyboard)
   - Review rule-based classification
   - View peri-saccade segment and features
   - Correct classification if wrong (click button or keyboard shortcut)
   - Add notes if needed
   - Save annotation (appends to master CSV)

3. **Efficiency Features:**
   - Filter by rule-based class to focus on specific types
   - Filter by confidence to prioritize low-confidence predictions
   - Sort by amplitude, time, etc.
   - Statistics: "X of Y saccades annotated"
   - Batch operations: "Accept all rule-based predictions" for high-confidence cases

4. **Output:**
   - Annotations saved to master CSV
   - Ready for ML training

### Verification Mode Workflow (Verify ML Predictions)

1. **Initial State:**
   - Load saccade results from current experiment
   - Extract features
   - Load trained ML model
   - Get ML predictions for all saccades
   - Get rule-based predictions (for comparison)
   - Display ML predictions as primary labels

2. **User Actions:**
   - Navigate through saccades
   - Review ML prediction and confidence
   - Compare with rule-based prediction (shown side-by-side)
   - Verify if ML prediction is correct
   - Correct if wrong (updates annotation)
   - Focus on disagreements (ML vs rule-based)
   - Focus on low-confidence ML predictions

3. **Efficiency Features:**
   - Filter by ML class
   - Filter by ML confidence (e.g., show only <0.7 confidence)
   - Filter by disagreements (ML ≠ rule-based)
   - Statistics: "ML accuracy: X%, disagreements: Y"
   - Color coding: Green (agreement), Red (disagreement), Yellow (low confidence)

4. **Output:**
   - Corrections saved to master CSV
   - Additional training data collected
   - Model performance metrics

### GUI Display Elements

**Saccade List Panel:**
- Columns: ID, Time, Amplitude, Duration, Current Label, Rule-Based Label, ML Label (if available), Confidence
- Color coding: Current label background color
- Highlight: Selected saccade row
- Filters: Dropdown menus for class, confidence range, amplitude range

**Time Series Plot:**
- Position trace (blue line)
- Velocity trace (red line)
- Saccades highlighted:
  - Current label color (if annotated)
  - Rule-based color (dashed, if different)
  - ML color (dotted, if different and available)
- Selected saccade: Thicker highlight

**Peri-Saccade Segment Plot:**
- Position aligned to threshold crossing
- Velocity aligned to threshold crossing
- Baseline window highlighted
- Saccade duration marked
- Feature values displayed as text

**Classification Panel:**
- Current label display (large, prominent)
- Rule-based prediction (smaller, with confidence)
- ML prediction (smaller, with confidence, if available)
- Classification buttons (1-4)
- Confidence slider (optional, for manual adjustment)
- Notes field (optional)

**Statistics Panel:**
- Training Mode: "X of Y annotated", "Progress: Z%"
- Verification Mode: "ML Accuracy: X%", "Disagreements: Y", "Low Confidence: Z"

## Questions to Consider

1. **Where to store models?**
   - Option A: `sleap/models/` (relative to code)
   - Option B: `data/models/` (with data)
   - **Recommendation**: `sleap/models/` for code portability

2. **Where to store annotations?**
   - Option A: `data/annotations/` (with data)
   - Option B: Project root `annotations/`
   - **Recommendation**: `data/annotations/` (with data, easier to backup)

3. **Model versioning strategy?**
   - Auto-increment (v1, v2, v3...)
   - Timestamp-based (v20250115_103000)
   - **Recommendation**: Auto-increment, simpler

4. **GUI complexity?**
   - Start minimal (MVP)
   - Add features iteratively
   - **Recommendation**: Start minimal, iterate

5. **Training frequency?**
   - After each annotation session
   - After N new annotations (e.g., 50)
   - Manual trigger
   - **Recommendation**: Manual trigger initially, auto-train later

6. **GUI mode selection?**
   - Automatic (detect if model exists)
   - Manual (user selects mode)
   - **Recommendation**: Automatic, but allow manual override

---

## Risk Mitigation

### Risk 1: TensorFlow MPS Issues on M2 Mac
- **Mitigation**: Test early, have CPU fallback
- **Alternative**: Use PyTorch if TensorFlow issues persist

### Risk 2: Insufficient Training Data
- **Mitigation**: Start with rule-based labels, collect annotations gradually
- **Strategy**: Use transfer learning or simpler model if needed

### Risk 3: GUI Complexity
- **Mitigation**: Start with minimal GUI, iterate based on feedback
- **Strategy**: Focus on core annotation workflow first

### Risk 4: Model Performance
- **Mitigation**: Use class weights, early stopping, validation
- **Strategy**: Iterate on features and architecture based on results

---

## Timeline Estimate

- **Week 1**: Core ML infrastructure + training pipeline
- **Week 2**: Model persistence + prediction functions + annotation storage
- **Week 3**: GUI development (basic version)
- **Week 4**: Integration + testing
- **Week 5+**: Refinement based on real data

**Total**: ~4-5 weeks for initial implementation

---

## Ready to Start?

**Recommended starting point**: Create `sleap/ml_classification.py` with TensorFlow backend setup and model building function. This establishes the foundation for everything else.

Would you like me to start implementing Phase 1 (Neural Network Model Architecture)?

