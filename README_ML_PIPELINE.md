# Universal ML Pipeline - Complete Package

## 📦 What's Included

### 1. Main Pipeline Script
**File:** `ml_pipeline.py`
- Universal script that works for all 4 datasets
- Implements complete leakage-safe ML pipeline
- 4 scalers × 8 algorithms = 32 experiments per dataset
- Comprehensive hyperparameter tuning
- 8 evaluation metrics per experiment

### 2. Usage Guide
**File:** `PIPELINE_USAGE_GUIDE.md`
- Quick start commands
- Detailed usage instructions
- Troubleshooting guide
- Parallel execution examples

### 3. Implementation Plan
**File:** `ML_PIPELINE_IMPLEMENTATION_PLAN.md`
- Complete architecture overview
- Methodology details
- Expected outcomes

### 4. Dependency Checker
**File:** `check_dependencies.py`
- Verifies all required packages
- Provides installation commands

---

## 🚀 Quick Start

### Step 1: Check Dependencies
```bash
python check_dependencies.py
```

### Step 2: Run Pipeline on Toddler Dataset
```bash
python ml_pipeline.py --dataset toddler --input Autism_Toddler_Data_Preprocessed.csv --output_dir results/toddler
```

### Step 3: Run with Overfitting Detection and Mitigation (Optional)
```bash
python ml_pipeline.py --dataset adult --input Autism_Adult_Data_Preprocessed.csv --output_dir results/adult --evaluate_train --apply_mitigation
```

---

## 📊 Pipeline Architecture

```
Input: Preprocessed CSV
    ↓
Train/Test Split (80/20, stratified)
    ↓
Leakage-Safe Encoding
    ├── One-Hot: gender, jaundice, family_asd, relation, ethnicity
    └── Target: contry_of_res (if present)
    ↓
For Each Scaler (4 total):
    ├── Quantile Transformer
    ├── Power Transformer
    ├── Normalizer
    └── Max Abs Scaler
        ↓
    SMOTE (training data only)
        ↓
    For Each Model (8 total):
        ├── Decision Tree (Grid Search)
        ├── KNN (Grid Search)
        ├── LDA (Grid Search)
        ├── Gaussian NB (Grid Search)
        ├── Logistic Regression (Bayesian)
        ├── AdaBoost (Bayesian)
        ├── Random Forest (Bayesian)
        └── SVM (Bayesian)
            ↓
        5-Fold CV Hyperparameter Tuning
            ↓
        Train on Balanced Data
            ↓
        Evaluate on Test Data
            ↓
        Calculate 8 Metrics:
            ├── Accuracy
            ├── ROC-AUC
            ├── F1-Score
            ├── Precision
            ├── Recall
            ├── MCC
            ├── Kappa
            └── Log Loss
    ↓
Save Results (CSV + JSON)
```

---

## 📈 Expected Output

### Per Dataset:
- **32 trained models** (4 scalers × 8 algorithms)
- **256 metric values** (32 models × 8 metrics)
- **Results CSV** with all experiments
- **Results JSON** with detailed information
- **Summary** showing best configuration

### Example Output Structure:
```
results/
└── toddler/
    ├── toddler_results.csv
    └── toddler_results.json
```

---

## ⏱️ Estimated Runtime

| Dataset | Samples | Estimated Time |
|---------|---------|----------------|
| Toddler | 1,054 | 3-4 hours |
| Adult | 704 | 2-3 hours |
| Child | 292 | 1-2 hours |
| Adolescent | 104 | 0.5-1 hour |

**Note:** Runtime depends on CPU cores and hyperparameter search iterations

---

## 🔑 Key Features

### ✅ Overfitting Detection & Mitigation
- **Training Metrics:** Use `--evaluate_train` to log training accuracy and detect 100% fit.
- **Automated Mitigation:** Use `--apply_mitigation` to automatically tighten hyperparameter grids (e.g., limit tree depth, regularization) to prevent overfitting.

### ✅ Leakage Prevention
- All transformations fit on **train data only**
- Test data never influences any preprocessing
- SMOTE applied to **train data only**

### ✅ Comprehensive Evaluation
- 8 metrics provide complete performance picture
- ROC-AUC used as primary optimization metric
- Results saved for later analysis

### ✅ Appropriate Tuning
- **Grid Search:** Simple models (DT, KNN, LDA, GNB)
- **Bayesian Optimization:** Moderate/complex models (LR, AB, RF, SVM)
- **5-fold CV:** All tuning methods

### ✅ Reproducibility
- Fixed random seed (42)
- Deterministic splits
- Consistent CV folds

### ✅ Input Validation & Missing Value Handling
- **Mandatory Parameters:** gender, jaundice, family_asd, contry_of_res, used_app_before
- **Optional Parameters:** age, ethnicity, relation (can have missing values)
- **'?' Handling:** Automatically replaced with mode from training data
- **NaN Handling:** Categorical NaN imputed with mode, numerical NaN imputed with median
- **Validation:** Checks for missing mandatory fields before processing

---

## 🔧 Customization

### Modify Hyperparameters
Edit `_get_param_grids()` in `ml_pipeline.py`

### Add/Remove Scalers
Edit `scalers` dictionary in `__init__()`

### Add/Remove Models
Edit `_get_models()` method

### Change CV Folds
Change `cv=5` to desired number in tuning methods

---

## 📝 For Other Agents

### To Run on Different Datasets:

**Adolescent:**
```bash
python ml_pipeline.py --dataset adolescent --input Autism_Adolescent_Data_Preprocessed.csv --output_dir results/adolescent
```

**Adult:**
```bash
python ml_pipeline.py --dataset adult --input Autism_Adult_Data_Preprocessed.csv --output_dir results/adult
```

**Child:**
```bash
python ml_pipeline.py --dataset child --input Autism_Child_Data_Preprocessed.csv --output_dir results/child
```

### Parallel Execution
Each agent can run their assigned dataset simultaneously in separate terminals/processes.

---

## 📚 Documentation Files

1. **PIPELINE_USAGE_GUIDE.md** - Detailed usage instructions
2. **ML_PIPELINE_IMPLEMENTATION_PLAN.md** - Architecture and methodology
3. **ENCODING_AND_SCALING_EXPLAINED.md** - Preprocessing details
4. **ENCODING_STRATEGY_ANALYSIS.md** - Encoding rationale
5. **DATASET_ANALYSIS_REPORT.md** - Initial data analysis
6. **CATEGORICAL_COLUMNS_ANALYSIS.md** - Feature analysis

---

## ✅ Ready to Execute

The pipeline is **fully implemented** and **ready to run** on any of the 4 datasets.

**Awaiting confirmation to execute on Toddler dataset.**
