"""
ML-Based Saccade Classification

This module implements Random Forest-based classification of saccades into four classes:
- Compensatory (0)
- Orienting / Purely Orienting (1)
- Saccade-and-Fixate (2)
- Non-Saccade (3)

Uses scikit-learn Random Forest Classifier with:
- Feature importance analysis
- Class weight balancing for imbalanced datasets
- Cross-platform support (M2 Mac, Ubuntu)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import pickle
from datetime import datetime

# Import scikit-learn components
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Install with: pip install scikit-learn")

# Import matplotlib for plotting (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib not available. Feature importance plotting will be disabled.")


# Class labels mapping
CLASS_LABELS = {
    0: 'compensatory',
    1: 'orienting',
    2: 'saccade_and_fixate',
    3: 'non_saccade'
}

CLASS_LABELS_REVERSE = {v: k for k, v in CLASS_LABELS.items()}


def build_random_forest_model(
    n_features: int,
    n_classes: int = 4,
    n_estimators: int = 200,
    max_depth: Optional[int] = 15,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: Union[str, int, float] = 'sqrt',
    class_weight: Union[str, Dict] = 'balanced',
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True
) -> Optional['RandomForestClassifier']:
    """
    Build a Random Forest Classifier for saccade classification.
    
    Random Forest is an ensemble of decision trees that:
    - Works excellently with tabular data
    - Provides feature importance scores
    - Handles non-linear relationships
    - Robust to outliers and missing values
    - Works well with small datasets (100-200 examples)
    
    Parameters
    ----------
    n_features : int
        Number of input features (for validation, not used in model)
    n_classes : int
        Number of output classes (default: 4)
    n_estimators : int
        Number of trees in the forest (default: 200)
    max_depth : int or None
        Maximum depth of trees (default: 15, None = unlimited)
    min_samples_split : int
        Minimum samples required to split a node (default: 5)
    min_samples_leaf : int
        Minimum samples required at a leaf node (default: 2)
    max_features : str, int, or float
        Number of features to consider for best split:
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - int: exact number
        - float: fraction of features
        (default: 'sqrt')
    class_weight : str or dict
        Weights for classes:
        - 'balanced': automatically adjust weights inversely proportional to class frequency
        - dict: {class_index: weight}
        (default: 'balanced')
    random_state : int
        Random seed for reproducibility (default: 42)
    n_jobs : int
        Number of parallel jobs (-1 = use all cores, default: -1)
    verbose : bool
        Whether to print model information
        
    Returns
    -------
    RandomForestClassifier or None
        Trained Random Forest model, or None if scikit-learn not available
    """
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not available. Cannot build model.")
        print("   Install with: pip install scikit-learn")
        return None
    
    # Build model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0  # sklearn verbose (0 = silent)
    )
    
    if verbose:
        print(f"\n📊 Random Forest Model Configuration:")
        print(f"   Input features: {n_features}")
        print(f"   Output classes: {n_classes}")
        print(f"   Number of trees: {n_estimators}")
        print(f"   Max depth: {max_depth if max_depth else 'unlimited'}")
        print(f"   Min samples split: {min_samples_split}")
        print(f"   Min samples leaf: {min_samples_leaf}")
        print(f"   Max features: {max_features}")
        print(f"   Class weight: {class_weight}")
        print(f"   Random state: {random_state}")
        print(f"   Parallel jobs: {n_jobs}")
    
    return model


def compute_class_weights(y_one_hot: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Uses sklearn's compute_class_weight with 'balanced' strategy.
    Useful for Random Forest's class_weight parameter.
    
    Parameters
    ----------
    y_one_hot : np.ndarray
        One-hot encoded labels, shape (n_samples, n_classes)
        
    Returns
    -------
    dict
        Dictionary mapping class index to weight
        Example: {0: 1.5, 1: 0.8, 2: 2.0, 3: 1.2}
    """
    if not SKLEARN_AVAILABLE:
        print("⚠️ sklearn not available. Using equal class weights.")
        n_classes = y_one_hot.shape[1]
        return {i: 1.0 for i in range(n_classes)}
    
    # Convert one-hot to class indices
    y_classes = np.argmax(y_one_hot, axis=1)
    
    # Compute balanced class weights
    classes = np.unique(y_classes)
    weights = compute_class_weight('balanced', classes=classes, y=y_classes)
    
    # Create dictionary
    class_weights = {int(cls): float(w) for cls, w in zip(classes, weights)}
    
    return class_weights


def encode_labels(labels: Union[List, np.ndarray, pd.Series], 
                  class_mapping: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode string labels to integers and create one-hot encoding.
    
    Parameters
    ----------
    labels : list, np.ndarray, or pd.Series
        String labels (e.g., ['compensatory', 'orienting', ...])
    class_mapping : dict, optional
        Mapping from string labels to integers. If None, creates mapping automatically.
        
    Returns
    -------
    tuple
        (y_one_hot, class_mapping)
        - y_one_hot: One-hot encoded labels, shape (n_samples, n_classes)
        - class_mapping: Dictionary mapping string labels to integers
    """
    labels = np.array(labels)
    
    # Create or use provided class mapping
    if class_mapping is None:
        unique_labels = np.unique(labels)
        class_mapping = {label: i for i, label in enumerate(sorted(unique_labels))}
    
    # Map labels to integers
    y_int = np.array([class_mapping[label] for label in labels])
    
    # One-hot encode
    n_classes = len(class_mapping)
    y_one_hot = np.zeros((len(y_int), n_classes))
    y_one_hot[np.arange(len(y_int)), y_int] = 1
    
    return y_one_hot, class_mapping


def decode_labels(y_one_hot: np.ndarray, 
                  class_mapping: Optional[Dict[int, str]] = None) -> np.ndarray:
    """
    Decode one-hot encoded labels back to string labels.
    
    Parameters
    ----------
    y_one_hot : np.ndarray
        One-hot encoded labels, shape (n_samples, n_classes)
    class_mapping : dict, optional
        Mapping from integers to string labels. If None, uses CLASS_LABELS.
        
    Returns
    -------
    np.ndarray
        String labels, shape (n_samples,)
    """
    if class_mapping is None:
        class_mapping = CLASS_LABELS
    
    # Convert one-hot to class indices
    y_int = np.argmax(y_one_hot, axis=1)
    
    # Map to string labels (handle both int and numpy int types)
    reverse_mapping = {int(k): v for k, v in class_mapping.items()}
    y_labels = np.array([reverse_mapping[int(i)] for i in y_int])
    
    return y_labels


def get_feature_importance(
    model: 'RandomForestClassifier',
    feature_names: List[str],
    normalize: bool = True
) -> pd.DataFrame:
    """
    Extract feature importance scores from a trained Random Forest model.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model
    feature_names : List[str]
        List of feature names (must match order of features used for training)
    normalize : bool
        Whether to normalize importance scores to sum to 1.0 (default: True)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance'], sorted by importance (descending)
        
    Raises
    ------
    ValueError
        If number of feature names doesn't match number of features in model
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
    
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model must be trained before extracting feature importance")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Validate feature names
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) doesn't match "
            f"number of features in model ({len(importances)})"
        )
    
    # Normalize if requested
    if normalize:
        importances = importances / importances.sum()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df


def analyze_feature_importance(
    model: 'RandomForestClassifier',
    feature_names: List[str],
    top_n: int = 20,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Analyze feature importance from a trained Random Forest model.
    
    Provides comprehensive statistics and analysis of feature importance.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model
    feature_names : List[str]
        List of feature names (must match order of features used for training)
    top_n : int
        Number of top features to highlight (default: 20)
    verbose : bool
        Whether to print analysis results
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'importance_df': pd.DataFrame, full feature importance DataFrame
        - 'top_features': pd.DataFrame, top N features
        - 'total_features': int, total number of features
        - 'max_importance': float, maximum importance score
        - 'mean_importance': float, mean importance score
        - 'std_importance': float, standard deviation of importance scores
        - 'cumulative_importance': pd.DataFrame, cumulative importance for top features
    """
    # Get feature importance DataFrame
    importance_df = get_feature_importance(model, feature_names, normalize=True)
    
    # Calculate statistics
    stats = {
        'importance_df': importance_df,
        'top_features': importance_df.head(top_n),
        'total_features': len(importance_df),
        'max_importance': importance_df['importance'].max(),
        'mean_importance': importance_df['importance'].mean(),
        'std_importance': importance_df['importance'].std(),
    }
    
    # Calculate cumulative importance
    top_n_df = importance_df.head(top_n).copy()
    top_n_df['cumulative_importance'] = top_n_df['importance'].cumsum()
    stats['cumulative_importance'] = top_n_df
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}")
        print(f"\nTotal features: {stats['total_features']}")
        print(f"Max importance: {stats['max_importance']:.4f}")
        print(f"Mean importance: {stats['mean_importance']:.4f}")
        print(f"Std importance: {stats['std_importance']:.4f}")
        
        print(f"\nTop {top_n} Features:")
        print(f"{'Rank':<6} {'Feature':<40} {'Importance':<12} {'Cumulative':<12}")
        print(f"{'-'*6} {'-'*40} {'-'*12} {'-'*12}")
        for idx, row in top_n_df.iterrows():
            rank = idx + 1
            feature = row['feature']
            importance = row['importance']
            cumulative = row['cumulative_importance']
            print(f"{rank:<6} {feature:<40} {importance:<12.4f} {cumulative:<12.4f}")
        
        # Show cumulative importance percentage
        cumulative_pct = top_n_df['cumulative_importance'].iloc[-1] * 100
        print(f"\nTop {top_n} features account for {cumulative_pct:.1f}% of total importance")
        
        print(f"\n{'='*80}\n")
    
    return stats


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True,
    save_path: Optional[Path] = None
) -> Optional['matplotlib.figure.Figure']:
    """
    Plot feature importance scores as a horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame from get_feature_importance() with columns ['feature', 'importance']
    top_n : int
        Number of top features to plot (default: 20)
    figsize : Tuple[int, int]
        Figure size (width, height) in inches (default: (10, 6))
    show_plot : bool
        Whether to display the plot (default: True)
    save_path : Path, optional
        Path to save the figure (default: None, don't save)
        
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object, or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ matplotlib not available. Cannot plot feature importance.")
        return None
    
    # Get top N features
    top_df = importance_df.head(top_n).copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart (reverse order so top feature is at top)
    y_pos = np.arange(len(top_df))
    ax.barh(y_pos, top_df['importance'], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df['feature'])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_df.iterrows()):
        importance = row['importance']
        ax.text(importance, i, f' {importance:.4f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig


# Test function
if __name__ == '__main__':
    print("Testing ML Classification Module (Random Forest)")
    print("=" * 60)
    
    if not SKLEARN_AVAILABLE:
        print("\n❌ scikit-learn not available. Install with: pip install scikit-learn")
        exit(1)
    
    # Test model building
    print("\n1. Testing Random Forest model building:")
    n_features = 36  # Number of features from feature extraction
    model = build_random_forest_model(n_features=n_features, n_classes=4, verbose=True)
    
    if model is not None:
        print("\n2. Testing class weights computation:")
        # Create dummy one-hot labels
        y_dummy = np.array([
            [1, 0, 0, 0],  # compensatory
            [1, 0, 0, 0],  # compensatory
            [0, 1, 0, 0],  # orienting
            [0, 0, 1, 0],  # saccade_and_fixate
            [0, 0, 0, 1],  # non_saccade
        ])
        weights = compute_class_weights(y_dummy)
        print(f"   Class weights: {weights}")
        
        print("\n3. Testing label encoding/decoding:")
        labels = ['compensatory', 'orienting', 'compensatory', 'non_saccade']
        y_one_hot, mapping = encode_labels(labels)
        print(f"   Labels: {labels}")
        print(f"   Mapping: {mapping}")
        print(f"   One-hot shape: {y_one_hot.shape}")
        # Create reverse mapping (int -> str)
        reverse_mapping = {int(v): str(k) for k, v in mapping.items()}
        decoded = decode_labels(y_one_hot, reverse_mapping)
        print(f"   Decoded: {decoded}")
        
        print("\n4. Testing feature importance functions:")
        # Create dummy feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Train model on dummy data (just to get feature importance)
        X_dummy = np.random.randn(100, n_features)
        y_dummy_classes = np.argmax(y_dummy, axis=1)
        # Repeat y_dummy_classes to match X_dummy shape
        y_dummy_classes_repeated = np.tile(y_dummy_classes, (100 // len(y_dummy_classes) + 1))[:100]
        model.fit(X_dummy, y_dummy_classes_repeated)
        
        # Test feature importance extraction
        importance_df = get_feature_importance(model, feature_names)
        print(f"   Extracted importance for {len(importance_df)} features")
        print(f"   Top 5 features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        # Test feature importance analysis
        stats = analyze_feature_importance(model, feature_names, top_n=10, verbose=True)
        
        # Test feature importance plotting (if matplotlib available)
        if MATPLOTLIB_AVAILABLE:
            print("\n5. Testing feature importance plotting:")
            fig = plot_feature_importance(importance_df, top_n=10, show_plot=False)
            if fig is not None:
                print("   ✅ Plot created successfully")
        else:
            print("\n5. Skipping plotting (matplotlib not available)")
        
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Model building failed")

