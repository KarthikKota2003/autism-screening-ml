# Universal ML Pipeline - Usage Guide

## Quick Start

### For Toddler Dataset:
```bash
python ml_pipeline.py --dataset toddler --input Autism_Toddler_Data_Preprocessed.csv --output_dir results/toddler
```

### For Other Datasets:
```bash
# Adolescent
python ml_pipeline.py --dataset adolescent --input Autism_Adolescent_Data_Preprocessed.csv --output_dir results/adolescent

# Adult
python ml_pipeline.py --dataset adult --input Autism_Adult_Data_Preprocessed.csv --output_dir results/adult

# Child
python ml_pipeline.py --dataset child --input Autism_Child_Data_Preprocessed.csv --output_dir results/child
```

---

## Pipeline Overview

### What It Does:
1. **Loads** preprocessed dataset
2. **Splits** train/test (80/20, stratified)
3. **Encodes** categorical features (leakage-safe)
   - One-Hot: gender, jaundice, family_asd, relation, ethnicity
   - Target: contry_of_res (if present)
4. **Scales** features using 4 methods:
   - Quantile Transformer
   - Power Transformer
   - Normalizer
   - Max Abs Scaler
5. **Balances** training data with SMOTE
6. **Trains** 8 ML models with hyperparameter tuning:
   - Decision Tree (Grid Search)
   - KNN (Grid Search)
   - LDA (Grid Search)
   - Gaussian NB (Grid Search)
   - Logistic Regression (Bayesian Optimization)
   - AdaBoost (Bayesian Optimization)
   - Random Forest (Bayesian Optimization)
   - SVM (Bayesian Optimization)
7. **Evaluates** with 8 metrics:
   - Accuracy
   - ROC-AUC
   - F1-Score
   - Precision
   - Recall
   - MCC
   - Kappa Score
   - Log Loss
8. **Saves** results to CSV and JSON

---

## Output Structure

```
results/
└── toddler/
    ├── toddler_results.csv      # All results in tabular format
    └── toddler_results.json     # All results in JSON format
```

### Results CSV Columns:
- dataset
- scaler
- model
- tuning_method
- best_params
- Accuracy
- ROC-AUC
- F1-Score
- Precision
- Recall
- MCC
- Kappa
- Log Loss

---

## Total Experiments Per Dataset

- **4 scalers** × **8 models** = **32 experiments**
- Each experiment includes:
  - 5-fold cross-validation for hyperparameter tuning
  - Training on balanced data (SMOTE)
  - Evaluation on original test data
  - 8 performance metrics

---

## Estimated Runtime

- **Simple models** (DT, KNN, LDA, GNB): ~2-5 min each
- **Moderate models** (LR, AB, RF): ~5-10 min each
- **Complex model** (SVM): ~10-20 min

**Total per dataset:** ~2-4 hours (depending on dataset size)

**Datasets by size:**
- Toddler: 1,054 samples (longest runtime)
- Adult: 704 samples
- Child: 292 samples
- Adolescent: 104 samples (shortest runtime)

---

## Requirements

Install required packages:
```bash
pip install pandas numpy scikit-learn imbalanced-learn category-encoders scikit-optimize scipy
```

---

## Key Features

### ✅ Leakage Prevention
- All encoders fit on **training data only**
- Scalers fit on **training data only**
- SMOTE applied to **training data only**
- Test data never influences any transformation

### ✅ Reproducibility
- Fixed random seed (42)
- Deterministic train/test split
- Consistent cross-validation folds

### ✅ Comprehensive Evaluation
- 8 metrics provide complete performance picture
- ROC-AUC used as primary metric for tuning
- Results saved for later analysis

### ✅ Appropriate Tuning
- Grid Search: Simple models (exhaustive search)
- Bayesian Optimization: Moderate/complex models (efficient search)
- 5-fold CV for all tuning

---

## Customization Options

### Change Random Seed:
```bash
python ml_pipeline.py --dataset toddler --input Autism_Toddler_Data_Preprocessed.csv --output_dir results/toddler --random_state 123
```

### Modify Hyperparameter Grids:
Edit the `_get_param_grids()` method in `ml_pipeline.py`

### Add More Scalers:
Edit the `scalers` dictionary in `__init__()`

### Add More Models:
Edit the `_get_models()` method

---

## Troubleshooting

### Error: "No module named 'skopt'"
```bash
pip install scikit-optimize
```

### Error: "No module named 'category_encoders'"
```bash
pip install category-encoders
```

### Memory Error:
- Reduce number of iterations in Bayesian Optimization
- Reduce number of folds in CV (change cv=5 to cv=3)

### Slow Runtime:
- Reduce hyperparameter search space
- Use fewer iterations for Bayesian Optimization
- Run on fewer scalers initially

---

## Parallel Execution for Multiple Datasets

You can run multiple datasets in parallel using different terminals/agents:

**Terminal 1:**
```bash
python ml_pipeline.py --dataset toddler --input Autism_Toddler_Data_Preprocessed.csv --output_dir results/toddler
```

**Terminal 2:**
```bash
python ml_pipeline.py --dataset adult --input Autism_Adult_Data_Preprocessed.csv --output_dir results/adult
```

**Terminal 3:**
```bash
python ml_pipeline.py --dataset child --input Autism_Child_Data_Preprocessed.csv --output_dir results/child
```

**Terminal 4:**
```bash
python ml_pipeline.py --dataset adolescent --input Autism_Adolescent_Data_Preprocessed.csv --output_dir results/adolescent
```

---

## Next Steps After Pipeline Completion

1. **Aggregate Results:**
   - Combine results from all datasets
   - Compare performance across age groups

2. **Analyze Best Configurations:**
   - Which scaler works best for each algorithm?
   - Which algorithm performs best overall?
   - Does optimal configuration vary by dataset?

3. **Visualize Results:**
   - Create comparison charts
   - Plot ROC curves
   - Generate confusion matrices

---

## Contact & Support

For issues or questions about the pipeline, refer to:
- `ML_PIPELINE_IMPLEMENTATION_PLAN.md` for detailed methodology
- `ENCODING_AND_SCALING_EXPLAINED.md` for preprocessing details
- `ENCODING_STRATEGY_ANALYSIS.md` for encoding rationale
