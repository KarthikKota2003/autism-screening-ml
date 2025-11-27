# Data Leakage Fix Report

## Executive Summary

**Critical data leakage discovered and fixed** in the autism screening ML pipeline. The `Qchat-10-Score` feature (sum of A1-A10 screening questions) was included in training data, allowing models to achieve artificially perfect predictions. This has been corrected by removing leakage features from all preprocessed datasets.

## Problem Discovered

### Initial Results (With Leakage)
When running the ML pipeline on the toddler dataset, **13 out of 32 experiments** achieved perfect scores:
- Accuracy: 1.0000
- ROC-AUC: 1.0000  
- F1-Score: 1.0000
- All other metrics: 1.0000

### Root Cause Analysis

The preprocessed data included two leakage features:

1. **`Qchat-10-Score`** - **PRIMARY LEAKAGE**
   - This is the sum of A1 through A10 (the 10 screening questions)
   - Directly correlates with the target variable `Class/ASD`
   - High scores (8-10) → almost always ASD diagnosis
   - Low scores (0-3) → almost always NO ASD
   - Models could trivially learn this relationship

2. **`Case_No`** - ID column with no predictive value

### Why This Caused Perfect Scores

Example from the data:
```
Row 2: Qchat-10-Score = 3  → Class/ASD = NO
Row 4: Qchat-10-Score = 10 → Class/ASD = YES  
Row 5: Qchat-10-Score = 10 → Class/ASD = YES
```

Complex models (DecisionTree, LogisticRegression, AdaBoost, RandomForest, SVM) easily learned:
- `if Qchat-10-Score >= threshold: predict YES else predict NO`

This defeats the entire purpose of building a predictive model from individual screening questions.

## Fix Applied

### Changes to Preprocessing Script

**File**: `preprocess_datasets.py`

Added **STEP 6: DROP LEAKAGE FEATURES** before saving datasets:

```python
# Features to drop:
# 1. Case_No / case_no - ID column with no predictive value
# 2. Qchat-10-Score - Sum of A1-A10, directly correlates with target (data leakage!)
leakage_features = ['Case_No', 'case_no', 'Qchat-10-Score', 'qchat-10-score']

for name, df in [('Adolescent', adolescent), ('Adult', adult), ('Child', child), ('Toddler', toddler)]:
    dropped = []
    for col in leakage_features:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            dropped.append(col)
```

### Verification Results

**Toddler Dataset:**
- ✅ `Case_No` removed
- ✅ `Qchat-10-Score` removed  
- ✅ Individual features A1-A10 retained
- ✅ Shape changed: 19 columns → 17 columns

**Other Datasets:**
- Adolescent, Adult, Child: No leakage features found (already clean)

## Next Steps

### Awaiting User Confirmation

Before re-running the pipeline, waiting for user to confirm:
1. Review this fix documentation
2. Approve re-execution of ML pipeline on toddler dataset

### Expected Results After Fix

When pipeline runs with corrected data:
- ❌ NO models should achieve perfect scores (1.0 across all metrics)
- ✅ Realistic performance: accuracy ~0.85-0.95, ROC-AUC ~0.90-0.98
- ✅ Variation across models and scalers
- ✅ Models learn from individual screening questions (A1-A10) only

### Comparison

| Metric | Before (With Leakage) | After (Expected) |
|--------|----------------------|------------------|
| Experiments with perfect scores | 13/32 (40.6%) | 0/32 (0%) |
| Best ROC-AUC | 1.0000 | ~0.92-0.98 |
| Model behavior | Trivial threshold on score | Learn patterns from A1-A10 |

## Lessons Learned

### For Future ML Projects

1. **Feature Engineering Review**: Always review derived features
   - Features that are combinations of other features
   - Features that directly encode the target
   
2. **Domain Knowledge**: Understanding what `Qchat-10-Score` represents was critical
   - It's a diagnostic tool output, not a predictor input
   
3. **Sanity Check Results**: Perfect scores are a red flag
   - Medical screening tasks rarely achieve 100% accuracy
   - Investigate when results seem "too good to be true"

4. **Data Preprocessing Audit**: 
   - Review all columns before training
   - Explicitly drop ID columns and derived scores
   - Keep only raw features that would be available at prediction time

## Technical Details

### Files Modified
- `preprocess_datasets.py` - Added leakage feature removal step
- All `*_Preprocessed.csv` files - Regenerated without leakage

### Files Ready for Re-execution
- `ml_pipeline.py` - No changes needed (already filters ID columns)
- `Autism_Toddler_Data_Preprocessed.csv` - Clean data ready

### Command to Re-run Pipeline
```bash
python ml_pipeline.py --dataset toddler --input Autism_Toddler_Data_Preprocessed.csv --output_dir results/toddler_corrected
```

---

**Status**: Fix implemented and verified. Awaiting user confirmation to re-execute pipeline.
