# Temporal Dynamics Analysis Guide
## Why Exponential Decay Fitting Fails & Better Alternatives

### Problem Summary

The exponential decay fit (`fit_exponential_decay`) in your turning analysis frequently fails (returns NaN) because:

#### 1. **Non-Exponential Dynamics**
Motor velocity during turning doesn't follow a simple exponential decay model: `y = A * exp(-t/τ)`.

**Why it fails:**
- The velocity may **stay elevated** (sustained turning response)
- The velocity may show **multi-phase dynamics** (fast initial decay, then plateau)
- The velocity may even **increase** after the initial peak
- Noise can break the exponential assumption

**Technical reason for NaN:**
The fit calculates: `slope = d(log(velocity))/dt`. If the slope ≥ 0, the signal is not decaying exponentially, so τ (tau) is undefined.

#### 2. **Biological Reality**
Your turning responses likely reflect:
- **Active compensatory behavior** (sustained effort, not passive decay)
- **Different strategies between halt and no-halt conditions**
- **Complex sensorimotor integration** (not a simple first-order system)

---

## Better Analysis Approaches

I've added three new sections to your script with more robust metrics:

### 1. **Windowed Area Under Curve (AUC)**

**What it measures:** Total "response magnitude" in different time windows

**Time windows:**
- `0-0.5s`: **Early response** - initial reaction magnitude
- `0.5-1s`: **Mid response** - sustained response
- `1-2s`: **Late response** - extended dynamics  
- `0-2s`: **Total response** - overall effect

**Why it's better:**
- ✅ Works with any curve shape (no assumptions about decay function)
- ✅ Captures both transient and sustained components
- ✅ Different windows reveal different phases of the response
- ✅ Robust to noise

**Interpretation:**
```python
If AUC_early(halt) < AUC_early(no_halt):
    → "Early turning response is reduced when visual feedback is halted"

If AUC_late(halt) > AUC_late(no_halt):
    → "Late-phase turning response is enhanced when visual feedback is halted"
```

### 2. **Sustained Response Ratio**

**What it measures:** `Ratio = |Velocity|_{1-2s} / |Velocity|_{0-0.5s}`

**Interpretation:**
- `Ratio < 1`: Response **decays** (transient)
- `Ratio ≈ 1`: Response is **sustained** (maintained)
- `Ratio > 1`: Response **increases** over time (builds up)

**Why it's useful:**
- ✅ Single number summarizing response dynamics
- ✅ Distinguishes transient vs sustained responses
- ✅ Directly comparable between conditions

**Example interpretation:**
```python
sustained_ratio(halt) = 0.8
sustained_ratio(no_halt) = 0.4

→ "Halt condition shows MORE sustained turning (less decay) compared to no-halt"
```

### 3. **Time to Baseline**

**What it measures:** How long it takes for velocity to return to pre-turn baseline levels (defined as baseline_mean + 2*SD)

**Why it's better than τ (tau):**
- ✅ Intuitive: "Response lasts ~1.5 seconds"
- ✅ Works even if decay is not exponential
- ✅ Captures the **functional duration** of the response

**Interpretation:**
```python
If time_to_baseline(halt) > time_to_baseline(no_halt):
    → "Turning responses last LONGER when visual feedback is halted"
```

---

## Usage Guide

### Running the New Analysis

The updated script automatically computes these metrics. Just run your normal analysis:

```python
# The script now includes:
# 1. Original analysis (with decay tau, often NaN)
# 2. Optional decay diagnostics (when enabled)
# 3. Alternative temporal metrics (always computed)
# 4. Statistical comparisons (paired t-tests)
# 5. Visualizations
```

### Enabling Decay Diagnostics (NEW!)

To understand why decay fits fail, enable diagnostics at the top of the script:

```python
# Configuration section (lines 69-71)
ENABLE_DECAY_DIAGNOSTICS = True   # Set to True to run diagnostics
MAX_DIAGNOSTIC_EXAMPLES = 3       # How many examples to diagnose in detail
```

**What this does:**
- Identifies all cases where decay tau = NaN
- Shows failure rate and distribution by group
- Runs detailed diagnostics on up to 3 examples
- Creates diagnostic plots showing:
  - Raw velocity trace
  - Log-linear plot (should be straight line for exponential decay)
  - Why the fit failed (slope, trend, etc.)
- Saves results to `decay_fit_diagnostics.csv`

**When to use:**
- First time running the analysis (to understand your data)
- If you see unexpected NaN patterns
- When debugging specific mice/conditions
- For teaching/presenting why exponential decay doesn't work

**Note:** This adds extra plots and output, so disable it (`False`) for routine analysis.

### Output Files (saved to your turning_analysis directory)

**Alternative metrics (always created):**
1. **`alternative_temporal_metrics.csv`** - Raw metrics per mouse
2. **`alternative_metrics_ttests.csv`** - Statistical comparisons
3. **`alternative_metrics_comparison.pdf`** - Visualization figure

**Diagnostics (only if enabled):**
4. **`decay_fit_diagnostics.csv`** - Detailed failure analysis per case

### Manual Diagnostic Tools (Advanced)

The diagnostic function is now integrated into the main script. For manual investigation, you can also call it directly:

```python
# In your script or a notebook, you can manually diagnose any trace:
from SANDBOX_4_MM_aligned_running_turning_analysis import diagnostic_exponential_fit

# Load your data
df = pd.read_csv("path/to/your/trace.csv")

# Run diagnostic on specific time window
tau, diagnostics = diagnostic_exponential_fit(
    df['Time (s)'].values, 
    df['Motor_Velocity'].values,
    title="Custom Diagnostic",
    show_plots=True
)

# Access diagnostic info
print(diagnostics['failure_reason'])
print(diagnostics['slope'])
```

The standalone `SANDBOX_4_decay_diagnostics.py` file also contains additional comprehensive analysis tools for exploratory analysis.

---

## Recommended Analysis Strategy

### For comparing halt vs no-halt conditions:

1. **Overall magnitude:** Use `AUC_0-2s` or `peak_velocity_abs_1s` (you already have this)

2. **Early vs late dynamics:** Compare `AUC_0-0.5s` vs `AUC_1-2s`
   - Tells you if one condition has a stronger initial response
   - Tells you if one condition maintains the response longer

3. **Response sustainability:** Use `sustained_ratio`
   - Higher ratio = more sustained response
   - Lower ratio = more transient response

4. **Response duration:** Use `time_to_baseline`
   - How long does the turning response last?

5. **Temporal profile:** Plot mean traces with SEM (you already have this)
   - Visual inspection of curve shapes

### Example Research Question Workflow

**Question:** "Does halting visual feedback change the temporal dynamics of turning?"

**Analysis:**
```python
# 1. Compare overall magnitude
→ Look at: AUC_0-2s comparison (paired t-test)

# 2. Test for differential effects over time
→ Look at: AUC_0-0.5s vs AUC_1-2s (interaction analysis)

# 3. Quantify sustainability
→ Look at: sustained_ratio comparison

# 4. Visualize
→ Plot: Mean traces + alternative metrics figure
```

**Interpretation Example:**
```
Results:
- AUC_0-2s: no difference (p=0.23)
- AUC_0-0.5s: halt < no-halt (p=0.02)
- AUC_1-2s: halt > no-halt (p=0.04)
- sustained_ratio: halt > no-halt (p=0.01)

Interpretation:
"While total turning magnitude was similar between conditions, 
halting visual feedback produced a more sustained turning response 
with reduced initial peak but prolonged late-phase activity."
```

---

## Technical Notes

### Why decay fitting is still in the code

The original `decay_tau` metric is still computed (for backward compatibility), but it will often be NaN. The new alternative metrics should be used instead.

### Statistical considerations

All metrics use:
- **Paired t-tests** (appropriate for within-subjects design)
- **Per-mouse averaging** (handles multiple trials per mouse)
- **SEM for error bars** (standard error of the mean across mice)

### Assumptions

The alternative metrics assume:
- **Monotonic or smooth velocity traces** (outliers can affect AUC)
- **Similar baseline periods** (-1 to 0s)
- **Aligned event timing** (time 0 = turn onset)

These are the same assumptions your existing analysis makes.

---

## Further Extensions

If you want even more detailed analysis, consider:

1. **Principal Component Analysis (PCA)** on time-series
   - Captures dominant temporal patterns
   - Can reveal distinct response modes

2. **Functional data analysis (FDA)**
   - Statistical framework specifically for comparing curves
   - Can test for differences at specific time points

3. **Mixed-effects models**
   - Account for mouse-to-mouse variability
   - Include covariates (session, day, etc.)

4. **Cross-correlation analysis**
   - Compare similarity of curve shapes
   - Detect temporal shifts

Let me know if you'd like help implementing any of these!

---

## Questions or Issues?

If the alternative metrics also show unexpected patterns:
1. Check raw traces visually (plot individual trials)
2. Verify time alignment (is t=0 correct?)
3. Check for outlier trials (extreme velocities)
4. Consider whether baseline subtraction is appropriate

The diagnostic script (`SANDBOX_4_decay_diagnostics.py`) has tools to help with all of these.

