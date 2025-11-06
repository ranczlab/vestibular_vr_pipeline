"""
ML-Based Saccade Classification

This module implements neural network-based classification of saccades into four classes:
- Compensatory (0)
- Orienting / Purely Orienting (1)
- Saccade-and-Fixate (2)
- Non-Saccade (3)

Uses TensorFlow/Keras with support for:
- M2 Mac: Metal Performance Shaders (MPS) backend
- Ubuntu: CPU or CUDA backend
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import pickle
from datetime import datetime

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")
    print("   For M2 Mac: pip install tensorflow-metal (optional, for MPS acceleration)")


# Class labels mapping
CLASS_LABELS = {
    0: 'compensatory',
    1: 'orienting',
    2: 'saccade_and_fixate',
    3: 'non_saccade'
}

CLASS_LABELS_REVERSE = {v: k for k, v in CLASS_LABELS.items()}


def setup_tensorflow_backend(verbose: bool = True) -> Dict[str, any]:
    """
    Setup TensorFlow backend (MPS for M2 Mac, CPU/CUDA for Ubuntu).
    
    Parameters
    ----------
    verbose : bool
        Whether to print backend information
        
    Returns
    -------
    dict
        Dictionary with backend information:
        - 'backend': str, backend name ('mps', 'gpu', 'cpu')
        - 'device': str, device name
        - 'available': bool, whether TensorFlow is available
    """
    if not TENSORFLOW_AVAILABLE:
        if verbose:
            print("❌ TensorFlow not available. Please install: pip install tensorflow")
        return {
            'backend': None,
            'device': None,
            'available': False
        }
    
    backend_info = {
        'available': True,
        'backend': None,
        'device': None
    }
    
    # Check for MPS (Metal Performance Shaders) on M2 Mac
    try:
        if hasattr(tf.config, 'list_physical_devices'):
            devices = tf.config.list_physical_devices()
            
            # Check for MPS
            mps_devices = [d for d in devices if 'mps' in d.name.lower()]
            if mps_devices:
                backend_info['backend'] = 'mps'
                backend_info['device'] = mps_devices[0].name
                if verbose:
                    print(f"✅ Using MPS (Metal) backend: {mps_devices[0].name}")
                return backend_info
            
            # Check for GPU (CUDA)
            gpu_devices = [d for d in devices if 'gpu' in d.name.lower()]
            if gpu_devices:
                backend_info['backend'] = 'gpu'
                backend_info['device'] = gpu_devices[0].name
                if verbose:
                    print(f"✅ Using GPU (CUDA) backend: {gpu_devices[0].name}")
                return backend_info
            
            # Fall back to CPU
            cpu_devices = [d for d in devices if 'cpu' in d.name.lower()]
            if cpu_devices:
                backend_info['backend'] = 'cpu'
                backend_info['device'] = cpu_devices[0].name if cpu_devices else 'CPU'
                if verbose:
                    print(f"✅ Using CPU backend: {backend_info['device']}")
                return backend_info
    except Exception as e:
        if verbose:
            print(f"⚠️ Error detecting backend: {e}")
    
    # Default to CPU if detection fails
    backend_info['backend'] = 'cpu'
    backend_info['device'] = 'CPU'
    if verbose:
        print(f"✅ Using CPU backend (default)")
    
    return backend_info


def build_mlp_model(
    n_features: int,
    n_classes: int = 4,
    hidden_units: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    activation: str = 'relu',
    use_batch_norm: bool = True,
    verbose: bool = True
) -> Optional['tf.keras.Model']:
    """
    Build a Multi-Layer Perceptron (MLP) model for saccade classification.
    
    Architecture:
    - Input Layer: (n_features,)
    - Hidden Layers: Dense + BatchNorm + Dropout + ReLU
    - Output Layer: Dense + Softmax (n_classes)
    
    Parameters
    ----------
    n_features : int
        Number of input features
    n_classes : int
        Number of output classes (default: 4)
    hidden_units : List[int]
        Number of units in each hidden layer (default: [128, 64, 32])
    dropout_rate : float
        Dropout rate for regularization (default: 0.3)
    activation : str
        Activation function for hidden layers (default: 'relu')
    use_batch_norm : bool
        Whether to use batch normalization (default: True)
    verbose : bool
        Whether to print model summary
        
    Returns
    -------
    tf.keras.Model or None
        Compiled Keras model, or None if TensorFlow not available
    """
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Cannot build model.")
        return None
    
    # Setup backend
    backend_info = setup_tensorflow_backend(verbose=verbose)
    
    # Build model
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(n_features,)))
    
    # Hidden layers
    for i, units in enumerate(hidden_units):
        # Dense layer
        model.add(layers.Dense(units, activation=activation, name=f'dense_{i+1}'))
        
        # Batch normalization (except for last hidden layer if only 1 layer)
        if use_batch_norm and (len(hidden_units) > 1 or i == 0):
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        
        # Dropout (reduce rate for later layers)
        current_dropout = dropout_rate if i == 0 else dropout_rate * 0.67
        model.add(layers.Dropout(current_dropout, name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(layers.Dense(n_classes, activation='softmax', name='output'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if verbose:
        print(f"\n📊 Model Architecture:")
        print(f"   Input features: {n_features}")
        print(f"   Hidden layers: {hidden_units}")
        print(f"   Output classes: {n_classes}")
        print(f"   Total parameters: {model.count_params():,}")
        model.summary()
    
    return model


def compute_class_weights(y_one_hot: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Uses sklearn's compute_class_weight with 'balanced' strategy.
    
    Parameters
    ----------
    y_one_hot : np.ndarray
        One-hot encoded labels, shape (n_samples, n_classes)
        
    Returns
    -------
    dict
        Dictionary mapping class index to weight
    """
    try:
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError:
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
    
    # Map to string labels
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    y_labels = np.array([reverse_mapping[i] for i in y_int])
    
    return y_labels


# Test function
if __name__ == '__main__':
    print("Testing ML Classification Module")
    print("=" * 60)
    
    # Test backend setup
    print("\n1. Testing TensorFlow backend setup:")
    backend_info = setup_tensorflow_backend(verbose=True)
    
    if backend_info['available']:
        # Test model building
        print("\n2. Testing model building:")
        n_features = 36  # Number of features from feature extraction
        model = build_mlp_model(n_features=n_features, n_classes=4, verbose=True)
        
        if model is not None:
            print("\n3. Testing class weights computation:")
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
            
            print("\n4. Testing label encoding/decoding:")
            labels = ['compensatory', 'orienting', 'compensatory', 'non_saccade']
            y_one_hot, mapping = encode_labels(labels)
            print(f"   Labels: {labels}")
            print(f"   Mapping: {mapping}")
            print(f"   One-hot shape: {y_one_hot.shape}")
            decoded = decode_labels(y_one_hot, {v: k for k, v in mapping.items()})
            print(f"   Decoded: {decoded}")
            
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Model building failed")
    else:
        print("\n⚠️ TensorFlow not available. Install with: pip install tensorflow")

