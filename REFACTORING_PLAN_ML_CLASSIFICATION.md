# Refactoring Plan: ml_classification.py

## Current State
- Uses TensorFlow/Keras for Neural Network
- Functions: `setup_tensorflow_backend()`, `build_mlp_model()`, `encode_labels()`, `decode_labels()`, `compute_class_weights()`
- Test function at bottom

## Target State
- Use scikit-learn Random Forest
- Keep same interface where possible
- Add feature importance analysis functions
- Remove TensorFlow dependency

## Function Mapping

### Keep (with minor updates)
- `encode_labels()` - Still needed for label encoding
- `decode_labels()` - Still needed for label decoding  
- `compute_class_weights()` - Still useful for RF (class_weight parameter)
- `CLASS_LABELS` - Keep as is

### Replace
- `setup_tensorflow_backend()` → Remove or make no-op (with deprecation warning)
- `build_mlp_model()` → `build_random_forest_model()`

### Add (New)
- `build_random_forest_model()` - Build RF classifier
- `get_feature_importance()` - Get feature importance scores
- `analyze_feature_importance()` - Comprehensive analysis
- `plot_feature_importance()` - Visualization

## Interface Design

### build_random_forest_model()
```python
def build_random_forest_model(
    n_features: int,
    n_classes: int = 4,
    n_estimators: int = 200,
    max_depth: Optional[int] = 15,
    min_samples_split: int = 5,
    class_weight: Union[str, Dict] = 'balanced',
    random_state: int = 42,
    verbose: bool = True
) -> RandomForestClassifier
```

Returns: sklearn RandomForestClassifier (not Keras model)

### get_feature_importance()
```python
def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str]
) -> pd.DataFrame
```

Returns: DataFrame with columns ['feature', 'importance'], sorted by importance

### analyze_feature_importance()
```python
def analyze_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
    top_n: int = 20,
    verbose: bool = True
) -> Dict
```

Returns: Dictionary with statistics and top features

### plot_feature_importance()
```python
def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True
) -> matplotlib.figure.Figure
```

Returns: matplotlib Figure

## Implementation Strategy

1. Keep imports minimal (scikit-learn, pandas, numpy, matplotlib)
2. Remove TensorFlow imports (with graceful handling)
3. Update docstrings to reflect Random Forest
4. Keep function signatures similar where possible
5. Update test function to test RF instead of MLP
6. Add comprehensive docstrings

## Testing

Test function should:
- Build RF model
- Test feature importance functions
- Test label encoding/decoding
- Test class weights

