# ML Saccade Classification - Implementation Summary

## Quick Reference: Key Decisions

### Classification Classes (4 distinct classes)
1. **Compensatory**: Saccades in bouts with slow eye movements between them
2. **Purely Orienting**: Freestanding saccades with stable gaze before/after  
3. **Saccade-and-Fixate**: First saccades of compensatory bouts (transitional type)
4. **Non-Saccade**: False positives that should be excluded from analysis

### Model Architecture: Neural Network (TensorFlow/Keras)
- **Type**: Multi-Layer Perceptron (MLP)
- **Platform Support**: 
  - M2 Mac: Metal Performance Shaders (MPS) backend
  - Ubuntu: CPU or CUDA (if available)
- **Architecture**: 2-3 hidden layers, 64-128 units per layer
- **Regularization**: Dropout (0.2-0.3), Batch Normalization
- **Class Handling**: Class weights for imbalanced data

### Annotation Workflow
1. **Initial**: Start with rule-based classifications pre-loaded in GUI
2. **User Action**: Review and correct classifications
3. **Storage**: Save all annotations to master CSV file (with experiment_id)
4. **Training**: Retrain model on all accumulated annotations (incremental training)
5. **Future Runs**: Use ML predictions, fallback to rule-based for low confidence

### Data Storage
- **Annotations**: Single master CSV file (`saccade_annotations_master.csv`)
  - Columns: `experiment_id`, `saccade_id`, `time`, `amplitude`, `duration`, `user_label`, `user_confidence`, `notes`, `annotation_date`
- **Models**: Versioned HDF5 files (`saccade_classifier_v1.h5`, `saccade_classifier_v2.h5`, etc.)
- **Metadata**: JSON files with model info, feature list, training stats

---

## Neural Network Architecture Details

### Recommended Architecture

```python
Model: Sequential
├── Input Layer: (n_features,)  # n_features = number of selected features
├── Dense Layer 1: 128 units, ReLU activation
│   ├── BatchNormalization
│   └── Dropout(0.3)
├── Dense Layer 2: 64 units, ReLU activation
│   ├── BatchNormalization
│   └── Dropout(0.2)
├── Dense Layer 3: 32 units, ReLU activation (optional)
│   └── Dropout(0.2)
└── Output Layer: 4 units, Softmax activation
    └── Classes: [compensatory, orienting, saccade_and_fixate, non_saccade]
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | Learning rate: 0.001 (default) |
| Loss Function | Categorical Crossentropy | For 4-class classification |
| Metrics | Accuracy, Per-class F1 | Track during training |
| Batch Size | 32-64 | Adjust based on dataset size |
| Epochs | 50-100 | Early stopping with patience=10 |
| Validation Split | 0.2 | 20% for validation |
| Class Weights | Computed | Balance imbalanced classes |

### M2 Mac Compatibility

**TensorFlow Setup:**
```python
import tensorflow as tf

# Check for MPS (Metal Performance Shaders) on M2 Mac
if tf.config.list_physical_devices('GPU'):
    print("Using GPU acceleration")
elif hasattr(tf.config, 'list_physical_devices') and 'mps' in str(tf.config.list_physical_devices()):
    # M2 Mac with MPS
    print("Using MPS (Metal) acceleration")
else:
    print("Using CPU")
```

**Ubuntu Compatibility:**
- Works with CPU backend (default)
- Can use CUDA if GPU available (optional)

---

## Incremental Training Strategy

### Workflow

1. **Initial Training** (First time):
   - Load annotations from master file
   - Extract features for all annotated saccades
   - Train model from scratch
   - Save as `saccade_classifier_v1.h5`

2. **Incremental Training** (After new annotations):
   - Load ALL annotations from master file (old + new)
   - Extract features for all experiments
   - Retrain model from scratch on full dataset
   - Save as `saccade_classifier_v2.h5` (increment version)
   - Keep old versions for comparison

3. **Why Retrain from Scratch?**
   - Simpler than fine-tuning
   - Avoids catastrophic forgetting
   - Better performance with full dataset
   - More predictable behavior

### Implementation

```python
def train_ml_classifier(annotations_file_path, features_dict, model_version=None):
    """
    Train ML classifier on all accumulated annotations.
    
    Parameters:
    -----------
    annotations_file_path : str
        Path to master annotations CSV file
    features_dict : dict
        Dictionary mapping experiment_id -> features DataFrame
        OR function to extract features on-the-fly
    model_version : int or None
        Version number for new model (auto-increment if None)
    
    Returns:
    --------
    model : tf.keras.Model
        Trained model
    training_stats : dict
        Training statistics (accuracy, loss, per-class metrics)
    """
    # 1. Load all annotations
    annotations_df = pd.read_csv(annotations_file_path)
    
    # 2. Get features for all annotated saccades
    # (Merge annotations with features by experiment_id + saccade_id)
    
    # 3. Prepare data
    X = features_array  # (n_samples, n_features)
    y = labels_one_hot  # (n_samples, 4)
    
    # 4. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Build model
    model = build_mlp_model(n_features=X_train_scaled.shape[1])
    
    # 7. Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y.argmax(axis=1))
    
    # 8. Train
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        class_weight=dict(enumerate(class_weights)),
        callbacks=[EarlyStopping(patience=10)]
    )
    
    # 9. Evaluate
    # ... compute metrics ...
    
    # 10. Save model and scaler
    model.save(f'saccade_classifier_v{model_version}.h5')
    save_scaler(scaler, f'scaler_v{model_version}.pkl')
    
    return model, training_stats
```

---

## GUI Design (Simple & Functional)

### Minimal Viable GUI

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Saccade Annotation Tool                                 │
├──────────────┬──────────────────────────────────────────┤
│              │  Time Series Plot                         │
│  Saccade     │  [Position + Velocity traces]            │
│  List        │                                          │
│  (scrollable)│  Peri-Saccade Segment                    │
│              │  [Aligned position/velocity]             │
│  Filters:    │                                          │
│  [Class ▼]   │  Classification:                         │
│  [Confidence]│  [1]Compensatory [2]Orienting            │
│              │  [3]Saccade-Fixate [4]Non-Saccade        │
│              │  [Save] [Next] [Previous]                │
└──────────────┴──────────────────────────────────────────┘
```

**Key Features:**
- Simple two-panel layout
- Saccade list on left (click to select)
- Plots on right (time series + peri-saccade segment)
- Keyboard shortcuts: 1-4 for classes, N/P for navigation, S for save
- Pre-populated with rule-based classifications
- Save annotations to master file

**Technology:** PyQt5 (simple, functional, cross-platform)

---

## Feature Selection Status

**Status**: ⏳ **Pending your selection**

See `FEATURE_SELECTION_LIST.md` for complete feature list organized by category.

**Recommended starting set**: ~30 features (see Feature Selection List for details)

---

## Next Steps

1. ✅ **Strategy Document**: Updated with your requirements
2. ✅ **Feature List**: Created comprehensive selection list
3. ⏳ **Feature Selection**: **Waiting for your input** - Please review `FEATURE_SELECTION_LIST.md` and select features
4. ⏳ **Implementation**: Once features selected, we'll implement:
   - Feature extraction function
   - Neural network model
   - Training pipeline
   - GUI annotation tool
   - Integration with existing notebook

---

## File Structure (Proposed)

```
sleap/
├── saccade_processing.py          # Existing (detection)
├── ml_classification.py            # NEW: ML model and training
├── feature_extraction.py           # NEW: Extract ML features
├── annotation_gui.py               # NEW: GUI for annotation
└── models/                         # NEW: Saved models
    ├── saccade_classifier_v1.h5
    ├── scaler_v1.pkl
    └── model_metadata_v1.json

data/
└── annotations/
    └── saccade_annotations_master.csv  # Master annotations file
```

---

## Questions?

If you have questions or need clarification on any aspect, please ask!

**Current Priority**: Feature selection - please review `FEATURE_SELECTION_LIST.md` and let me know which features to include.

