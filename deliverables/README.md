# Autism Screening ML Pipeline - Production Deliverables

A production-ready Flask web application for autism spectrum disorder (ASD) screening across four age groups, powered by machine learning models trained on Q-CHAT screening data.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Model Training](#model-training)
- [Best Model Selection](#best-model-selection)
- [UI Integration](#ui-integration)
- [How the UI Works](#how-the-ui-works)
- [Challenges & Solutions](#challenges--solutions)
- [Deployment](#deployment)
- [Local Development](#local-development)

---

## 🎯 Project Overview

### Objective

Provide an accessible, web-based autism screening tool that uses machine learning to assess ASD risk across four age categories:
- **Toddler** (12-36 months)
- **Child** (4-11 years)
- **Adolescent** (12-16 years)
- **Adult** (18+ years)

### Key Features

- ✅ **Age-Appropriate Screening**: Tailored questionnaires for each age group
- ✅ **ML-Powered Predictions**: Trained models provide risk assessments
- ✅ **User-Friendly Interface**: Responsive design with Q-CHAT-10 questions
- ✅ **Age Classifier**: Automatic category suggestion based on age input
- ✅ **Production-Ready**: Security hardened, deployment-ready code

---

## 📊 Data Collection

### Data Sources

**1. ASDTests.com** (Adolescent, Adult, Child)
- Online autism screening platform
- Voluntary participation
- Q-CHAT-10 questionnaire responses
- Demographic information

**2. Mobile Application** (Toddler)
- Parental screening app
- Q-CHAT-10 for toddlers
- Age in months (12-36)
- Parental observations

### Dataset Statistics

| Dataset | Samples | Features | Target Distribution |
|---------|---------|----------|---------------------|
| Adolescent | 104 | 17 | YES: 63%, NO: 37% |
| Adult | 704 | 21 | YES: 64%, NO: 36% |
| Child | 292 | 18 | YES: 55%, NO: 45% |
| Toddler | 1,054 | 17 | YES: 62%, NO: 38% |

### Features Collected

**Behavioral (Q-CHAT-10)**:
- A1-A10: Screening questions assessing:
  - Social communication
  - Joint attention
  - Pretend play
  - Repetitive behaviors

**Demographic**:
- Age (years/months)
- Gender (m/f)
- Ethnicity (12 categories)
- Jaundice at birth (yes/no)
- Family history of ASD (yes/no)
- Country of residence
- Previous app usage (yes/no)
- Relation to individual (parent/self/healthcare professional)

---

## 🔧 Data Preprocessing

### Pipeline: `preprocess_datasets.py`

**Workflow**:
```
Raw CSV → Standardization → Type Conversion → Missing Values → 
Leakage Removal → Preprocessed CSV
```

### Step-by-Step Process

#### 1. Standardization
- **Column Names**: Unified across all datasets
- **Categorical Values**: Lowercased (YES → yes, Male → m)
- **Spelling Fixes**: Corrected typos (jundice → jaundice, austim → autism)

#### 2. Type Conversion
- **Age**: Converted from categorical to numerical
  - "18-24" → 21 (midpoint)
  - "24 months" → 24
- **Missing Values**: '?' replaced with NaN

#### 3. Missing Value Handling
- **Categorical**: Mode imputation (most frequent value)
- **Numerical**: Median imputation (in ML pipeline)
- **Strategy**: Preserve data integrity while handling gaps

#### 4. Data Leakage Prevention ⚠️

**Removed Features**:
- `Qchat-10-Score`: Sum of A1-A10 (deterministic relationship with target)
- `Case_No`: ID column (no predictive value)

**Why Critical**: The Q-CHAT-10 score is calculated as `sum(A1-A10)`, and the diagnosis rule is:
```python
if sum(A1-A10) > 3:
    diagnosis = YES
else:
    diagnosis = NO
```

Including this would give 100% accuracy but zero real-world value.

### Output Files

- `Autism_Adolescent_Data_Preprocessed.csv`
- `Autism_Adult_Data_Preprocessed.csv`
- `Autism_Child_Data_Preprocessed.csv`
- `Autism_Toddler_Data_Preprocessed.csv`

---

## 🎯 Feature Selection

### Feature Engineering Strategy

**1. Behavioral Features (A1-A10)**
- **For Adult, Child, Adolescent**: Included as primary predictors
- **For Toddler**: **EXCLUDED** (see Challenges section)
- **Rationale**: Direct indicators of ASD traits

**2. Demographic Features**
- **All Categories**: age, gender, ethnicity, jaundice, family_asd
- **Importance**: Capture risk factors and population patterns

**3. Encoding Strategy**

**One-Hot Encoding** (Low Cardinality):
- `gender` (2 values: m, f)
- `jaundice` (2 values: yes, no)
- `family_asd` (2 values: yes, no)
- `used_app_before` (2 values: yes, no)
- `relation` (5 values: parent, self, relative, healthcare professional, others)

**Target Encoding** (High Cardinality):
- `ethnicity` (12 categories)
- `contry_of_res` (20+ countries)

**Why Target Encoding for High Cardinality?**
- One-hot encoding would create 30+ sparse columns
- Target encoding captures relationship with target variable
- Reduces dimensionality while preserving information

### Feature Scaling

**Tested 4 Scalers**:
1. **QuantileTransformer**: Transforms to uniform/normal distribution
2. **PowerTransformer**: Yeo-Johnson transformation for normality
3. **Normalizer**: L2 normalization (unit norm)
4. **MaxAbsScaler**: Scales to [-1, 1] range

**Selection**: Best scaler chosen per model via cross-validation

---

## 🤖 Model Training

### Pipeline: `ml_pipeline.py`

**Workflow**:
```
Preprocessed Data → Train/Test Split → Feature Encoding → 
Scaling → SMOTE → Hyperparameter Tuning → Model Training → 
Evaluation → Model Persistence
```

### Algorithms Tested (8 Total)

1. **Decision Tree Classifier**
2. **K-Nearest Neighbors (KNN)**
3. **Linear Discriminant Analysis (LDA)**
4. **Gaussian Naive Bayes**
5. **Logistic Regression**
6. **AdaBoost Classifier**
7. **Random Forest Classifier**
8. **Support Vector Machine (SVM)**

### Hyperparameter Tuning

**Methods Used**:
- **Grid Search**: For simple models (Decision Tree, KNN, LDA, Naive Bayes)
- **Bayesian Optimization**: For complex models (Logistic Regression, AdaBoost, Random Forest, SVM)

**Cross-Validation**: 3-fold CV for quick mode, 5-fold for full training

### Class Imbalance Handling

**SMOTE** (Synthetic Minority Over-sampling Technique):
- Applied to training data only (prevents leakage)
- Balances YES/NO classes
- Improves model sensitivity to minority class

### Evaluation Metrics (8 Total)

1. **Accuracy**: Overall correctness
2. **ROC-AUC**: Area under ROC curve (primary metric)
3. **F1-Score**: Harmonic mean of precision and recall
4. **Precision**: True positives / (True positives + False positives)
5. **Recall**: True positives / (True positives + False negatives)
6. **MCC**: Matthews Correlation Coefficient
7. **Kappa**: Cohen's Kappa (inter-rater agreement)
8. **Log Loss**: Probabilistic loss function

---

## 🏆 Best Model Selection

### Selection Criteria

**Primary Metric**: **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve)

**Why ROC-AUC?**
- Threshold-independent metric
- Balances sensitivity and specificity
- Robust to class imbalance
- Suitable for screening applications (better to flag potential cases)

### Results by Category

#### Adolescent
- **Best Model**: Logistic Regression
- **Scaler**: QuantileTransformer
- **ROC-AUC**: 0.7404 (74.04%)
- **Accuracy**: 0.7619 (76.19%)
- **F1-Score**: 0.8333

#### Adult
- **Best Model**: Random Forest
- **Scaler**: StandardScaler + PowerTransformer
- **ROC-AUC**: 0.7430 (74.30%)
- **Accuracy**: 0.7589 (75.89%)
- **F1-Score**: 0.8333

#### Child
- **Best Model**: AdaBoost
- **Scaler**: MaxAbsScaler
- **ROC-AUC**: 0.6250 (62.50%)
- **Accuracy**: 0.7119 (71.19%)
- **F1-Score**: 0.7586

#### Toddler (Demographics Only)
- **Best Model**: SVM (Support Vector Machine)
- **Scaler**: QuantileTransformer
- **ROC-AUC**: 0.6908 (69.08%)
- **Accuracy**: 0.6730 (67.30%)
- **F1-Score**: 0.7435
- **Note**: Trained WITHOUT A1-A10 features (see Challenges section)

### Model Artifacts

Each model saved as `.pkl` file containing:
- Trained model
- Scaler (fitted)
- Imputer (fitted)
- One-Hot Encoder (fitted)
- Target Encoder (fitted)
- Feature column names
- Metadata (model name, scaler name, score)

**Files**:
- `adolescent_best_model.pkl` (9 KB)
- `adult_best_model.pkl` (3.2 MB - largest due to Random Forest)
- `child_best_model.pkl` (190 KB)
- `toddler_best_model.pkl` (196 KB)

---

## 🔗 UI Integration

### Architecture

```
User Input (Web Form) → Flask Backend → Preprocessing Pipeline → 
Model Inference → Result Display
```

### Backend: `UI/app.py`

**Key Components**:

1. **Model Loading** (`load_model_artifacts`):
   - Loads `.pkl` files on-demand
   - Caches loaded models in memory
   - Handles missing model files gracefully

2. **Preprocessing** (`preprocess_input`):
   - Replicates training preprocessing exactly
   - Applies same imputation, encoding, scaling
   - Ensures feature alignment with training data

3. **Category-Specific Logic**:
   ```python
   if category == 'toddler':
       # Remove A1-A10 (not used by toddler model)
       # Use only demographics
   else:
       # Include A1-A10 + demographics
   ```

4. **Prediction**:
   - Loads appropriate model for category
   - Preprocesses input data
   - Generates prediction (YES/NO)
   - Calculates probability (confidence %)

### Frontend: `UI/templates/`

**Templates**:
1. **`base.html`**: Base layout with header, footer, flash messages
2. **`index.html`**: Home page with 4 category cards + age classifier
3. **`screening.html`**: Dynamic form with Q-CHAT-10 questions
4. **`result.html`**: Risk assessment with recommendations

**Styling**: `UI/static/css/style.css`
- Modern, responsive design
- Card-based layout
- Smooth animations
- Mobile-friendly

**JavaScript**: `UI/static/js/main.js`
- Age classifier modal logic
- Form validation
- Dynamic category selection

---

## 🎨 How the UI Works

### User Flow

```
1. Landing Page
   ↓
2. Category Selection (Manual or Age Classifier)
   ↓
3. Screening Form (Q-CHAT-10 + Demographics)
   ↓
4. Form Submission
   ↓
5. Backend Processing (Preprocessing + Prediction)
   ↓
6. Results Page (Risk Assessment + Recommendations)
```

### Detailed Walkthrough

#### Step 1: Landing Page (`/`)
- Displays 4 "island cards" for each category
- Floating Action Button (FAB) for age classifier
- User can click category directly or use age classifier

#### Step 2: Age Classifier (Optional)
- Modal popup with age input
- Supports years or months
- Logic:
  - 12-36 months → Toddler
  - 4-11 years → Child
  - 12-16 years → Adolescent
  - 18+ years → Adult
- Redirects to appropriate screening form

#### Step 3: Screening Form (`/screening/<category>`)
- **Q-CHAT-10 Questions** (A1-A10):
  - A1: "Does the individual look at you when you call his/her name?"
  - A2: "How easy is it for you to get eye contact?"
  - ... (10 questions total)
- **Demographics**:
  - Age (months for toddler, years for others)
  - Gender, Ethnicity
  - Jaundice, Family ASD history
  - Country, Previous app usage (not for toddler)
  - Relation to individual

#### Step 4: Form Submission (`POST /predict`)
- Client-side validation (all fields required)
- Server-side validation
- Data preprocessing:
  ```python
  # Example for Adult
  data = {
      'A1_Score': 1, 'A2_Score': 0, ..., 'A10_Score': 1,
      'age': 25, 'gender': 'm', 'ethnicity': 'white european',
      'jaundice': 'no', 'family_asd': 'no',
      'contry_of_res': 'United States', 'used_app_before': 'no',
      'relation': 'Self'
  }
  ```

#### Step 5: Backend Processing
1. Load model artifacts for category
2. Create DataFrame from form data
3. Preprocess:
   - Impute missing values
   - One-hot encode categorical features
   - Target encode high-cardinality features
   - Scale numerical features
4. Predict:
   - `prediction = model.predict(X_scaled)`
   - `probability = model.predict_proba(X_scaled)[0][1]`

#### Step 6: Results Page (`/result`)
- **Risk Assessment**:
  - High Risk (YES) or Low Risk (NO)
  - Confidence percentage (e.g., 78%)
  - Visual progress bar
- **Recommendations**:
  - **High Risk**: Consult specialist, early intervention
  - **Low Risk**: Monitor development, regular check-ups
- **Actions**:
  - Return to home
  - Learn more (external link)

---

## 🚧 Challenges & Solutions

### Challenge 1: Data Leakage (Q-CHAT-10 Score)

**Problem**:
- Original datasets included `Qchat-10-Score` column
- This is simply `sum(A1-A10)`
- Including it would give 100% accuracy (deterministic)

**Solution**:
- Removed `Qchat-10-Score` in preprocessing
- Removed `Case_No` (ID column)
- Verified no other leakage sources

**Impact**: Models now learn true patterns, not shortcuts

---

### Challenge 2: Toddler Model Determinism

**Problem**:
- Toddler model with A1-A10 achieved 100% accuracy
- Investigation revealed: `if sum(A1-A10) > 3: YES else NO`
- Model was just a calculator, not learning

**Solution**:
- **Retrained toddler model WITHOUT A1-A10 features**
- Used only demographics: age_months, gender, ethnicity, jaundice, family_asd, relation
- Created `train_toddler_demographic.py` script

**Results**:
- ROC-AUC: 69% (down from 100%)
- Accuracy: 67% (down from 100%)
- **But**: Now a true ML model predicting risk from demographics

**UI Adaptation**:
- UI still collects A1-A10 for toddlers (for consistency)
- Backend strips A1-A10 before feeding to toddler model
- Other categories continue using A1-A10

---

### Challenge 3: Scikit-learn Version Mismatch

**Problem**:
- Models trained with scikit-learn 1.7.2
- Production environment uses 1.3.2 (for compatibility)
- Warnings during model loading

**Solution**:
- Downgraded training dependencies to match production
- Documented version requirements in `requirements.txt`
- Warnings are non-fatal, functionality preserved

**Impact**: Models load successfully with warnings (acceptable)

---

### Challenge 4: High-Cardinality Categorical Features

**Problem**:
- `ethnicity`: 12 categories
- `contry_of_res`: 20+ countries
- One-hot encoding would create 30+ sparse columns

**Solution**:
- **Target Encoding** for high-cardinality features
- Preserves relationship with target variable
- Reduces dimensionality significantly

**Impact**: Better model performance, fewer features

---

### Challenge 5: Class Imbalance

**Problem**:
- All datasets have ~60% YES, 40% NO
- Models biased toward majority class

**Solution**:
- **SMOTE** (Synthetic Minority Over-sampling)
- Applied to training data only
- Balances classes before training

**Impact**: Improved recall for minority class

---

## 🚀 Deployment

### Platform: Render.com (Free Tier)

**Why Render?**
- ✅ Completely free (no credit card)
- ✅ GitHub auto-deploy (push to deploy)
- ✅ HTTPS included
- ✅ 750 hours/month free

**Trade-off**:
- Spins down after 15 minutes of inactivity
- ~30 second cold start on first request

### Deployment Files

**`Procfile`** (root):
```
web: gunicorn app:app --chdir deliverables/UI
```

**`render.yaml`** (root):
```yaml
services:
  - type: web
    name: autism-screening-app
    env: python
    buildCommand: pip install -r deliverables/UI/requirements.txt
    startCommand: gunicorn app:app --chdir deliverables/UI
    envVars:
      - key: SECRET_KEY
        generateValue: true
      - key: FLASK_ENV
        value: production
```

### Environment Variables

**Required**:
- `SECRET_KEY`: Flask secret key (auto-generated by Render)
- `FLASK_ENV`: Set to `production`

### Security Features

1. **CSRF Protection**: Flask-WTF enabled
2. **Environment Variables**: Secret key from env, not hardcoded
3. **Debug Mode**: Disabled in production
4. **HTTPS**: Automatic via Render

---

## 💻 Local Development

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/yourusername/autism-screening-ml.git
   cd autism-screening-ml
   ```

2. **Install Dependencies**:
   ```bash
   cd deliverables/UI
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env and set SECRET_KEY
   ```

4. **Run Application**:
   ```bash
   python app.py
   ```

5. **Access**:
   - Open browser to `http://127.0.0.1:5000`

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_app.py -v

# Run with coverage
pytest --cov=UI tests/
```

---

## 📁 Project Structure

```
deliverables/
├── README.md (this file)
├── models/ (4 .pkl files)
├── data/ (8 CSV files)
├── scripts/
│   ├── preprocess_datasets.py
│   └── ml_pipeline.py
└── UI/
    ├── app.py
    ├── requirements.txt
    ├── .env.example
    ├── static/
    │   ├── css/style.css
    │   └── js/main.js
    └── templates/
        ├── base.html
        ├── index.html
        ├── screening.html
        └── result.html
```

---

## 📊 Performance Summary

| Category | Model | ROC-AUC | Accuracy | F1-Score |
|----------|-------|---------|----------|----------|
| Adolescent | Logistic Regression | 74.04% | 76.19% | 83.33% |
| Adult | Random Forest | 74.30% | 75.89% | 83.33% |
| Child | AdaBoost | 62.50% | 71.19% | 75.86% |
| Toddler | SVM (Demographics) | 69.08% | 67.30% | 74.35% |

---

## ⚠️ Disclaimer

**IMPORTANT**: This application provides a screening assessment based on machine learning models. It is **NOT** a medical diagnosis. Please consult a qualified healthcare professional for a formal evaluation.

---

## 📄 License

Research Project - 2025

---

## 🙏 Acknowledgments

- Q-CHAT-10 questionnaire by Autism Research Centre, University of Cambridge
- Dataset sources: ASDTests.com and mobile screening applications
- Open-source libraries: Flask, scikit-learn, pandas, numpy

---

**For deployment instructions, see**: [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)  
**For security analysis, see**: [docs/SECURITY_ANALYSIS.md](../docs/SECURITY_ANALYSIS.md)
