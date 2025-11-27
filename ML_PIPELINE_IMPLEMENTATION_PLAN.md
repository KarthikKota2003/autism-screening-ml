# ML Pipeline Implementation Plan

## Overview
Implement a comprehensive, leakage-safe machine learning pipeline for autism screening across 4 datasets using 4 scaling methods, 8 ML algorithms, sophisticated hyperparameter tuning, and 8 evaluation metrics.

---

## Architecture

### Pipeline Structure
```
For each dataset (Adolescent, Adult, Child, Toddler):
  ├── Train/Test Split (80/20, stratified)
  ├── Encoding (leakage-safe)
  │   ├── One-Hot: gender, jaundice, family_asd, relation, ethnicity
  │   ├── Target: contry_of_res (if present)
  │   └── Label: Class/ASD → 0/1
  ├── For each Scaler (QT, PT, Normalizer, MAS):
  │   ├── Fit on train, transform train & test
  │   ├── Apply SMOTE on train only
  │   └── For each Algorithm (DT, KNN, LDA, GNB, LR, AB, RF, SVM):
  │       ├── Hyperparameter tuning (method depends on algorithm)
  │       ├── Train on balanced data
  │       ├── Predict on test data
  │       └── Calculate 8 metrics
  └── Aggregate results
```

**Total Experiments:** 4 datasets × 4 scalers × 8 algorithms = **128 model configurations**

---

## Phase 1: Leakage-Safe Preprocessing

### 1.1 Data Loading
- Load 4 preprocessed datasets
- Verify no missing values
- Confirm standardized column names

### 1.2 Train/Test Split
```python
from sklearn.model_selection import train_test_split

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 1.3 Feature Encoding

#### One-Hot Encoding (Low Cardinality)
```python
from sklearn.preprocessing import OneHotEncoder

ohe_cols = ['gender', 'jaundice', 'family_asd', 'relation', 'ethnicity']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(X_train[ohe_cols])  # Fit on train only!
```

#### Target Encoding (High Cardinality)
```python
from category_encoders import TargetEncoder

te_cols = ['contry_of_res']  # Only in Adolescent, Adult, Child
te = TargetEncoder(cols=te_cols, smoothing=1.0)
te.fit(X_train[te_cols], y_train)  # Fit on train only!
```

#### Label Encoding (Target)
```python
y_train = (y_train == 'YES').astype(int)  # YES=1, NO=0
y_test = (y_test == 'YES').astype(int)
```

---

## Phase 2: Feature Scaling Methods

### 2.1 Quantile Transformer (QT)
```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal', random_state=42)
```
- **Purpose:** Transforms features to follow normal distribution
- **Best for:** Non-linear transformations, handling outliers
- **Output:** Gaussian distribution

### 2.2 Power Transformer (PT)
```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson', standardize=True)
```
- **Purpose:** Makes data more Gaussian-like using power transformations
- **Best for:** Stabilizing variance, making data more normal
- **Methods:** Yeo-Johnson (handles zero and negative values)

### 2.3 Normalizer
```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')
```
- **Purpose:** Scales individual samples to unit norm
- **Best for:** Text classification, when direction matters more than magnitude
- **Note:** Scales rows, not columns!

### 2.4 Max Abs Scaler (MAS)
```python
from sklearn.preprocessing import MaxAbsScaler

mas = MaxAbsScaler()
```
- **Purpose:** Scales by maximum absolute value
- **Best for:** Sparse data, preserves zero entries
- **Output:** Range [-1, 1]

---

## Phase 3: Class Balancing with SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
```

**Apply AFTER scaling, BEFORE training!**

---

## Phase 4: ML Algorithms

### Simple Models

#### 4.1 Decision Tree (DT)
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
```

#### 4.2 K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
```

#### 4.3 Linear Discriminant Analysis (LDA)
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
```

#### 4.4 Gaussian Naive Bayes (GNB)
```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
```

### Moderate Models

#### 4.5 Logistic Regression (LR)
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42)
```

#### 4.6 AdaBoost (AB)
```python
from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier(random_state=42)
```

#### 4.7 Random Forest (RF)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
```

### Complex Model

#### 4.8 Support Vector Machine (SVM)
```python
from sklearn.svm import SVC

svm = SVC(probability=True, random_state=42)
```

---

## Phase 5: Hyperparameter Tuning

### 5.1 Grid Search (Simple Models: DT, KNN, LDA, GNB)

```python
from sklearn.model_selection import GridSearchCV

# Decision Tree
dt_params = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# KNN
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# LDA
lda_params = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
}

# GNB
gnb_params = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
```

### 5.2 Random Search → Bayesian (Moderate: LR, AB, RF)

```python
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from scipy.stats import uniform, randint

# Logistic Regression
lr_params = {
    'C': uniform(0.01, 10),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# AdaBoost
ab_params = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 1.0)
}

# Random Forest
rf_params = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Random Search first (faster exploration)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

# Then Bayesian Optimization (refined search)
bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=params,
    n_iter=30,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
```

### 5.3 Bayesian Optimization (Complex: SVM)

```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

svm_params = {
    'C': Real(0.1, 100, prior='log-uniform'),
    'gamma': Real(1e-4, 1e-1, prior='log-uniform'),
    'kernel': Categorical(['rbf', 'poly', 'sigmoid']),
    'degree': Categorical([2, 3, 4])  # For poly kernel
}

bayes_search = BayesSearchCV(
    estimator=SVC(probability=True, random_state=42),
    search_spaces=svm_params,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
```

---

## Phase 6: Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss
)

def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba[:, 1]),
        'F1-Score': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'Log Loss': log_loss(y_true, y_pred_proba)
    }
    return metrics
```

---

## Phase 7: Results Organization

### Output Structure
```
results/
├── adolescent/
│   ├── quantile_transformer/
│   │   ├── dt_results.json
│   │   ├── knn_results.json
│   │   └── ... (8 algorithms)
│   ├── power_transformer/
│   ├── normalizer/
│   └── max_abs_scaler/
├── adult/
├── child/
├── toddler/
└── summary/
    ├── best_models_per_dataset.csv
    ├── best_scalers_per_algorithm.csv
    └── overall_rankings.csv
```

---

## Implementation Timeline

1. **Phase 1-2:** Preprocessing & Scaling setup (~30 min)
2. **Phase 3:** SMOTE implementation (~10 min)
3. **Phase 4:** Model training loop (~20 min)
4. **Phase 5:** Hyperparameter tuning (~2-4 hours runtime)
5. **Phase 6:** Metrics calculation (~10 min)
6. **Phase 7:** Results aggregation (~20 min)

**Total Development Time:** ~2 hours
**Total Execution Time:** ~4-6 hours (due to hyperparameter tuning)

---

## Key Design Decisions

### Why This Approach?

1. **Leakage Prevention:** All transformations fit on train only
2. **Comprehensive Comparison:** 4 scalers × 8 algorithms = robust insights
3. **Appropriate Tuning:** Match tuning complexity to model complexity
4. **Rich Metrics:** 8 metrics provide complete performance picture
5. **Reproducibility:** Fixed random seeds throughout

### Expected Insights

- Which scaler works best for each algorithm?
- Which algorithm performs best on each dataset?
- Does scaling method matter more for certain algorithms?
- How do age groups (toddler/child/adolescent/adult) affect model performance?

---

## Next Steps

Ready to implement! This will create:
- ✅ 128 trained models (4 datasets × 4 scalers × 8 algorithms)
- ✅ 1,024 metric values (128 models × 8 metrics)
- ✅ Comprehensive comparison tables
- ✅ Best model recommendations per dataset

Proceed with implementation?
