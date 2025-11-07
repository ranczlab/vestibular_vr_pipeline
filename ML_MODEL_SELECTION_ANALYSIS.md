# ML Model Selection Analysis for Saccade Classification

## Task Characteristics

### Classification Problem
- **4 Classes**: Compensatory, Orienting, Saccade-and-Fixate, Non-Saccade
- **Data Type**: Tabular features (not time series sequences)
- **Feature Count**: ~36 features
- **Dataset Size**: Initially 100-200 labeled examples, growing incrementally
- **Feature Types**: 
  - Continuous (amplitude, velocity, drift, variance, etc.)
  - Categorical/Ordinal (direction, bout position, is_first_in_bout, etc.)
  - Temporal (time intervals, bout duration)
  - Statistical (means, variances, ratios)

### Key Requirements
1. **Interpretability**: Need to understand which features matter
2. **Debugging**: If model doesn't work well, need actionable ways to improve
3. **Small Dataset**: Must work with limited labeled data initially
4. **Incremental Learning**: Should improve as more annotations are added
5. **Feature Importance**: Need to know which features are most discriminative
6. **Robustness**: Handle outliers, missing values, imbalanced classes

---

## Model Options Analysis

### Option 1: Random Forest Classifier ⭐ **RECOMMENDED**

**Why it fits:**
- **Excellent for tabular data**: Random Forest is specifically designed for structured/tabular data
- **Feature importance**: Provides clear feature importance scores
- **Robust**: Handles outliers, missing values, non-linear relationships
- **No scaling needed**: Works with raw features
- **Works with small datasets**: Can train on 100-200 examples
- **Interpretable**: Can inspect trees, see decision paths
- **No TensorFlow dependency**: Uses scikit-learn (already in your environment)

**Architecture:**
- Ensemble of decision trees (100-200 trees)
- Each tree votes, majority wins
- Built-in feature importance calculation
- Handles imbalanced classes with class_weight='balanced'

**If it doesn't work well:**
1. **Analyze feature importance** → Identify which features matter most
2. **Add/remove features** → Focus on important ones, remove noise
3. **Tune hyperparameters** → max_depth, n_estimators, min_samples_split
4. **Collect more data** → Especially for underrepresented classes
5. **Feature engineering** → Create new features based on insights
6. **Class weights** → Adjust for imbalanced classes

**Pros:**
- ✅ No TensorFlow dependency (uses scikit-learn)
- ✅ Feature importance analysis
- ✅ Works well with small datasets
- ✅ Robust to outliers and noise
- ✅ Handles non-linear relationships
- ✅ Easy to debug and improve
- ✅ Fast training and prediction
- ✅ Cross-platform (works on M2 Mac, Ubuntu)

**Cons:**
- ⚠️ May not capture very complex feature interactions
- ⚠️ Less "black box" than neural networks, but still ensemble

**Implementation Complexity:** Low (scikit-learn, ~50 lines of code)

---

### Option 2: Gradient Boosting (XGBoost or LightGBM)

**Why it fits:**
- **Best performance** for tabular data (often outperforms RF)
- **Feature importance**: Provides feature importance scores
- **Handles imbalanced data**: Built-in class weights
- **Fast training**: Especially LightGBM
- **No TensorFlow dependency**: Separate package

**If it doesn't work well:**
- Similar to Random Forest: feature importance, hyperparameter tuning, more data

**Pros:**
- ✅ Often best accuracy for tabular data
- ✅ Feature importance available
- ✅ Fast training (LightGBM)
- ✅ Handles imbalanced data well
- ✅ No TensorFlow dependency

**Cons:**
- ⚠️ More hyperparameter tuning needed
- ⚠️ Less interpretable than Random Forest
- ⚠️ Requires separate package installation
- ⚠️ More sensitive to hyperparameters

**Implementation Complexity:** Medium (need to install XGBoost/LightGBM, ~100 lines)

---

### Option 3: Logistic Regression (Baseline)

**Why consider it:**
- **Simple baseline**: Good starting point
- **Highly interpretable**: Can see exact feature coefficients
- **Fast**: Very fast training and prediction
- **Linear relationships**: Assumes linear decision boundaries

**If it doesn't work well:**
- Limited - can only capture linear relationships
- Would need to move to non-linear model

**Pros:**
- ✅ Very interpretable (feature coefficients)
- ✅ Fast training
- ✅ Good baseline
- ✅ No dependencies beyond scikit-learn

**Cons:**
- ❌ Assumes linear relationships (may not fit saccade data)
- ❌ Limited flexibility
- ❌ May not capture complex patterns

**Implementation Complexity:** Very Low (~20 lines)

---

### Option 4: Support Vector Machine (SVM)

**Why consider it:**
- **Non-linear**: Can use RBF kernel for non-linear boundaries
- **Works with small datasets**: Good for 100-200 examples
- **Robust**: Handles outliers well

**If it doesn't work well:**
- Limited debugging options
- Harder to interpret than tree-based methods

**Pros:**
- ✅ Handles non-linear relationships (with RBF kernel)
- ✅ Works with small datasets
- ✅ Robust to outliers

**Cons:**
- ⚠️ Less interpretable
- ⚠️ Slower training than RF/XGBoost
- ⚠️ Requires feature scaling
- ⚠️ Harder to debug

**Implementation Complexity:** Medium (~50 lines)

---

### Option 5: Neural Network (TensorFlow) - Current Choice

**Why it was chosen:**
- Can learn complex patterns
- Handles non-linear relationships
- Scalable to large datasets

**Why reconsider:**
- ❌ TensorFlow compatibility issues with SLEAP
- ❌ Requires ARM64 Python (environment complexity)
- ❌ Less interpretable (harder to debug)
- ❌ Requires more data (typically needs 1000+ examples)
- ❌ If it doesn't work: Limited debugging options (black box)

**If it doesn't work well:**
- Hard to debug (black box)
- Can try: More data, different architecture, hyperparameter tuning
- But less actionable than tree-based methods

**Pros:**
- ✅ Can learn very complex patterns
- ✅ Scalable to large datasets

**Cons:**
- ❌ TensorFlow dependency conflicts
- ❌ Less interpretable
- ❌ Requires more data
- ❌ Harder to debug
- ❌ Environment complexity (ARM64 Python)

**Implementation Complexity:** High (TensorFlow setup, ~200+ lines)

---

## Recommendation: Random Forest ⭐

### Why Random Forest is Best for This Task

1. **Perfect for Tabular Data**
   - Your features are tabular (not time series sequences)
   - Random Forest excels at structured data
   - No need for neural network complexity

2. **Feature Importance Analysis**
   - Provides clear feature importance scores
   - Can identify which features matter most
   - Helps with feature engineering

3. **Debugging & Improvement Path**
   - If accuracy is low → Check feature importance → Add/remove features
   - If certain classes fail → Analyze trees → Understand decision paths
   - If imbalanced → Adjust class weights
   - Very actionable improvement path

4. **Small Dataset Friendly**
   - Works well with 100-200 examples
   - Doesn't require thousands of samples like neural networks

5. **No Dependency Conflicts**
   - Uses scikit-learn (already in your environment)
   - No TensorFlow/SLEAP conflicts
   - Works on both M2 Mac and Ubuntu

6. **Incremental Learning**
   - Can retrain on expanded dataset easily
   - Fast training (seconds, not minutes)

### Implementation Strategy

**Phase 1: Start with Random Forest**
- Train on initial 100-200 annotations
- Analyze feature importance
- Evaluate performance

**Phase 2: If Needed, Try Gradient Boosting**
- If RF accuracy is good but want to squeeze more → Try XGBoost/LightGBM
- Compare performance

**Phase 3: Neural Network (Optional, Later)**
- Only if tree-based methods plateau
- After collecting 1000+ annotations
- When TensorFlow environment is stable

---

## Comparison Table

| Model | Accuracy | Interpretability | Small Data | Debugging | Dependencies | Complexity |
|-------|----------|------------------|------------|-----------|--------------|------------|
| **Random Forest** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | scikit-learn | Low |
| XGBoost/LightGBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | xgboost/lightgbm | Medium |
| Logistic Regression | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | scikit-learn | Very Low |
| SVM | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | scikit-learn | Medium |
| Neural Network | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | TensorFlow | High |

---

## Specific to Your Saccade Features

### Feature Characteristics That Favor Random Forest:

1. **Mixed Feature Types**: RF handles continuous, categorical, and ordinal features well
2. **Feature Interactions**: RF captures interactions through tree splits
3. **Non-linear Relationships**: RF handles non-linear patterns (e.g., amplitude vs duration)
4. **Feature Importance**: Critical for understanding which features distinguish classes
   - Example: "Is pre_saccade_drift more important than bout_size?"
5. **Outliers**: RF is robust to outliers (important for saccade data)
6. **Missing Values**: Can handle missing values (though your features are complete)

### Example Feature Importance Analysis:

With Random Forest, you'll get output like:
```
Feature Importance:
- pre_saccade_position_drift: 0.15
- bout_size: 0.12
- post_saccade_position_variance: 0.11
- amplitude: 0.10
- time_since_previous_saccade: 0.08
...
```

This tells you:
- Which features matter most
- Which features to focus on for improvement
- Which features might be redundant

---

## If Random Forest Doesn't Work Well

### Actionable Improvement Path:

1. **Analyze Feature Importance**
   ```python
   feature_importance = model.feature_importances_
   # Identify low-importance features → Remove or engineer better ones
   ```

2. **Per-Class Analysis**
   ```python
   # Check which features distinguish each class
   # Focus on features that separate problematic classes
   ```

3. **Hyperparameter Tuning**
   ```python
   # Tune: n_estimators, max_depth, min_samples_split
   # Use GridSearchCV or RandomizedSearchCV
   ```

4. **Feature Engineering**
   ```python
   # Based on importance analysis, create new features
   # Example: ratio features, interaction terms
   ```

5. **Collect More Data**
   ```python
   # Focus on underrepresented classes
   # Collect examples where model is uncertain
   ```

6. **Class Weights**
   ```python
   # Adjust for imbalanced classes
   class_weight='balanced' or custom weights
   ```

7. **Try Gradient Boosting**
   ```python
   # If RF plateaus, try XGBoost/LightGBM
   # Often 5-10% better accuracy
   ```

---

## Final Recommendation

**Start with Random Forest** because:

1. ✅ **No TensorFlow conflicts** - Uses scikit-learn only
2. ✅ **Feature importance** - Understand what matters
3. ✅ **Works with small data** - 100-200 examples is enough
4. ✅ **Easy to debug** - Clear improvement path
5. ✅ **Fast** - Trains in seconds
6. ✅ **Interpretable** - Can inspect trees
7. ✅ **Robust** - Handles your feature types well

**If RF works well (85%+ accuracy):** Great! Stick with it.

**If RF needs improvement:** 
- Analyze feature importance → Engineer better features
- Tune hyperparameters
- Collect more data (especially edge cases)
- Try XGBoost/LightGBM for 5-10% boost

**If RF plateaus:** Then consider Neural Network (after collecting 1000+ examples)

---

## Implementation Plan (Random Forest)

### Step 1: Replace Neural Network with Random Forest
- Update `ml_classification.py` to use Random Forest
- Keep same interface (train, predict, save/load)
- Add feature importance analysis

### Step 2: Training Pipeline
- Use scikit-learn's RandomForestClassifier
- Cross-validation for evaluation
- Feature importance visualization

### Step 3: Model Persistence
- Save as pickle (simpler than HDF5)
- Save feature importance with model
- Save metadata (training stats, feature list)

### Step 4: Prediction & Debugging
- Predict function
- Feature importance analysis function
- Per-class performance analysis

---

## Code Example (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Train
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Decision Framework

**Choose Random Forest if:**
- ✅ You want interpretability and debugging capability
- ✅ You have 100-200+ labeled examples
- ✅ You want to avoid TensorFlow complexity
- ✅ You need feature importance analysis
- ✅ You want fast iteration and improvement

**Choose XGBoost/LightGBM if:**
- ✅ Random Forest works but you want 5-10% more accuracy
- ✅ You're willing to tune more hyperparameters
- ✅ You can install additional packages

**Choose Neural Network if:**
- ✅ You have 1000+ labeled examples
- ✅ Random Forest/XGBoost plateau
- ✅ You're willing to deal with TensorFlow setup
- ✅ You need maximum accuracy (and can debug black box)

---

## My Strong Recommendation

**Start with Random Forest** - It's the sweet spot for your use case:
- Perfect for tabular data
- Feature importance for debugging
- Works with small datasets
- No dependency conflicts
- Easy to improve iteratively

You can always upgrade to XGBoost or Neural Network later if needed, but Random Forest will likely work very well and be much easier to work with.

**Would you like me to:**
1. Refactor `ml_classification.py` to use Random Forest instead?
2. Keep the same interface so it's easy to switch models later?
3. Add feature importance analysis functions?

