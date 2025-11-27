"""
Autism Dataset Preprocessing Script
====================================

Purpose:
--------
This script preprocesses 4 autism screening datasets (Adolescent, Adult, Child, Toddler)
to prepare them for machine learning pipeline training. It standardizes column names,
handles missing values, fixes data quality issues, and removes data leakage features.

Workflow:
---------
1. Load all 4 raw datasets
2. Standardize capitalization and spelling across datasets
3. Convert age columns to numerical format
4. Handle missing values ('?' and NaN)
5. Perform mean/mode imputation for missing values
6. Drop redundant columns (age_desc)
7. Remove data leakage features (Qchat-10-Score, Case_No)
8. Save preprocessed datasets

Key Features:
-------------
- Unified preprocessing for all 4 age groups
- Data leakage prevention (removes Qchat-10-Score which is sum of A1-A10)
- Consistent column naming across datasets
- Missing value imputation strategies
- Data quality validation

Output:
-------
- Autism_Adolescent_Data_Preprocessed.csv
- Autism_Adult_Data_Preprocessed.csv
- Autism_Child_Data_Preprocessed.csv
- Autism_Toddler_Data_Preprocessed.csv
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

print("="*80)
print("AUTISM DATASET PREPROCESSING")
print("="*80)

# ============================================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================================
# Load the 4 raw autism screening datasets from CSV files
# Each dataset targets a different age group with slightly different column names
print("\n[1/7] Reading datasets...")
adolescent = pd.read_csv('Autism_Adolescent_Data.csv')  # Ages 12-16
adult = pd.read_csv('Autism_Adult_Data.csv')            # Ages 18+
child = pd.read_csv('Autism_Child_Data.csv')            # Ages 4-11
toddler = pd.read_csv('Autism_Toddler_Data.csv')        # Ages 12-36 months

print(f"  Adolescent: {adolescent.shape}")
print(f"  Adult: {adult.shape}")
print(f"  Child: {child.shape}")
print(f"  Toddler: {toddler.shape}")

# ============================================================================
# STEP 2: STANDARDIZE CAPITALIZATION AND SPELLING
# ============================================================================
# Problem: Different datasets use inconsistent capitalization and spelling
# Solution: Standardize all categorical values to lowercase and fix typos
# This ensures consistent encoding in the ML pipeline
print("\n[2/7] Standardizing capitalization and spelling...")

def standardize_values(df, column_mapping):
    """
    Utility function to standardize categorical values across datasets
    
    Args:
        df: DataFrame to standardize
        column_mapping: Dict mapping column names to value replacement dicts
    
    Returns:
        DataFrame with standardized values
    """
    for col, value_map in column_mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(value_map)
    return df

# Standardize gender/sex column
# Issue: Toddler dataset uses 'Sex', others use 'gender'
# Issue: Mixed case values (M/m, F/f)
# Fix: Rename to 'gender' and convert to lowercase
for df in [adolescent, adult, child]:
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.lower()  # m/f

if 'Sex' in toddler.columns:
    toddler['Sex'] = toddler['Sex'].str.lower()
    toddler.rename(columns={'Sex': 'gender'}, inplace=True)  # Unify column name

print("  OK Gender standardized to lowercase (m/f)")

# Standardize ethnicity column
# Issue: Inconsistent capitalization and spelling variations
# Examples: 'White-European' vs 'White European', 'Pacifica' vs 'Pasifika'
# Fix: Map all variations to lowercase standardized values
ethnicity_mapping = {
    'White-European': 'white european',
    'White European': 'white european',
    'Middle Eastern ': 'middle eastern',  # Note: trailing space in original data
    'Middle Eastern': 'middle eastern',
    'South Asian': 'south asian',
    'south asian': 'south asian',
    'Asian': 'asian',
    'Black': 'black',
    'Hispanic': 'hispanic',
    'Latino': 'latino',
    'Others': 'others',
    'Turkish': 'turkish',
    'Pasifika': 'pasifika',
    'Pacifica': 'pasifika',  # Spelling variation
    'Native Indian': 'native indian',
    'mixed': 'mixed'
}

for df in [adolescent, adult, child]:
    if 'ethnicity' in df.columns:
        df['ethnicity'] = df['ethnicity'].replace(ethnicity_mapping)

if 'Ethnicity' in toddler.columns:
    toddler['Ethnicity'] = toddler['Ethnicity'].replace(ethnicity_mapping)
    toddler.rename(columns={'Ethnicity': 'ethnicity'}, inplace=True)

print("  OK Ethnicity standardized to lowercase")

# Fix spelling error: jundice -> jaundice
# Issue: Original datasets have typo 'jundice' instead of 'jaundice'
# Fix: Rename column and standardize to lowercase
for df in [adolescent, adult, child]:
    if 'jundice' in df.columns:
        df.rename(columns={'jundice': 'jaundice'}, inplace=True)
        df['jaundice'] = df['jaundice'].str.lower()  # yes/no

if 'Jaundice' in toddler.columns:
    toddler.rename(columns={'Jaundice': 'jaundice'}, inplace=True)
    toddler['jaundice'] = toddler['jaundice'].str.lower()

print("  OK Fixed spelling: jundice -> jaundice")

# Standardize autism family history column
# Issue: Column name varies ('austim' vs 'Family_mem_with_ASD')
# Issue: Typo 'austim' instead of 'autism'
# Fix: Rename all to 'family_asd' and standardize to lowercase
for df in [adolescent, adult, child]:
    if 'austim' in df.columns:
        df.rename(columns={'austim': 'family_asd'}, inplace=True)
        df['family_asd'] = df['family_asd'].str.lower()  # yes/no

if 'Family_mem_with_ASD' in toddler.columns:
    toddler.rename(columns={'Family_mem_with_ASD': 'family_asd'}, inplace=True)
    toddler['family_asd'] = toddler['family_asd'].str.lower()

print("  OK Standardized family ASD history column")

# Standardize relation/who completed test column
# Issue: Different column names and inconsistent capitalization
# Issue: Similar categories need consolidation (e.g., 'family member' -> 'parent')
# Fix: Unify column name to 'relation' and standardize values
relation_mapping = {
    'Health care professional': 'health care professional',
    'Health Care Professional': 'health care professional',
    'family member': 'parent',  # Consolidate: family member is typically parent
    'Parent': 'parent',
    'Relative': 'relative',
    'Self': 'self',
    'Others': 'others'
}

for df in [adolescent, adult, child]:
    if 'relation' in df.columns:
        df['relation'] = df['relation'].replace(relation_mapping)

if 'Who completed the test' in toddler.columns:
    toddler.rename(columns={'Who completed the test': 'relation'}, inplace=True)
    toddler['relation'] = toddler['relation'].replace(relation_mapping)

print("  OK Standardized relation column")

# Standardize target variable (Class/ASD)
# Issue: Inconsistent capitalization (Yes/YES/yes, No/NO/no)
# Fix: Convert all to uppercase (YES/NO) for consistency
for df in [adolescent, adult, child]:
    if 'Class/ASD' in df.columns:
        df['Class/ASD'] = df['Class/ASD'].str.upper()

if 'Class/ASD Traits ' in toddler.columns:
    toddler.rename(columns={'Class/ASD Traits ': 'Class/ASD'}, inplace=True)
    toddler['Class/ASD'] = toddler['Class/ASD'].str.upper()

print("  OK Standardized target variable to uppercase (YES/NO)")

# Standardize used_app_before column
for df in [adolescent, adult, child]:
    if 'used_app_before' in df.columns:
        df['used_app_before'] = df['used_app_before'].str.lower()

print(\"  OK Standardized used_app_before column\")

# ============================================================================
# STEP 3: CONVERT AGE TO NUMERICAL
# ============================================================================
# Problem: Age is stored as categorical/string in some datasets
# Problem: Missing values represented as '?' instead of NaN
# Solution: Convert to numerical format and handle missing values
print(\"\\n[3/7] Converting age to numerical...\")

# Adult dataset - age is stored as string
if 'age' in adult.columns:
    # First replace '?' with NaN for proper handling
    adult['age'] = adult['age'].replace('?', np.nan)
    # Convert to numeric (errors='coerce' converts invalid values to NaN)
    adult['age'] = pd.to_numeric(adult['age'], errors='coerce')
    print(f\"  OK Adult: Converted age to numerical (missing values: {adult['age'].isnull().sum()})\")

# Child dataset - same issue as adult
if 'age' in child.columns:
    child['age'] = child['age'].replace('?', np.nan)
    child['age'] = pd.to_numeric(child['age'], errors='coerce')
    print(f\"  OK Child: Converted age to numerical (missing values: {child['age'].isnull().sum()})\")

# Adolescent dataset - doesn't have age column, only age_desc (categorical)
# Solution: Create age column with median value for age group (12-16)
if 'age' not in adolescent.columns:
    adolescent['age'] = 14  # Median of 12-16
    print(f\"  OK Adolescent: Created age column with median value (14)\")

# Toddler dataset - age is in months, not years
# Rename for clarity to avoid confusion with age in years
if 'Age_Mons' in toddler.columns:
    toddler.rename(columns={'Age_Mons': 'age_months'}, inplace=True)
    print(f"  OK Toddler: Renamed Age_Mons to age_months")

# ============================================================================
# STEP 4: HANDLE '?' AS MISSING VALUES
# ============================================================================
# Problem: Missing values are represented as '?' strings in categorical columns
# Solution: Replace '?' with NaN for proper missing value handling
print("\n[4/7] Handling '?' as missing values...")

missing_counts = {}
for name, df in [('Adolescent', adolescent), ('Adult', adult), ('Child', child), ('Toddler', toddler)]:
    # Replace '?' with NaN in all object columns
    for col in df.select_dtypes(include=['object']).columns:
        before = (df[col] == '?').sum()
        if before > 0:
            df[col] = df[col].replace('?', np.nan)
            print(f"  {name}.{col}: {before} '?' values replaced with NaN")
            missing_counts[f"{name}.{col}"] = before

if not missing_counts:
    print("  OK No '?' values found in any dataset")

# ============================================================================
# STEP 5: MISSING VALUE IMPUTATION
# ============================================================================
# Strategy: Use mode (most frequent value) for categorical columns
# Rationale: Mode imputation preserves the distribution of categorical data
# Note: Numerical columns will be handled later in the ML pipeline
print("\n[5/7] Performing Mean Value Imputation...")

# First, identify numerical columns (we'll skip imputation here as ML pipeline handles it)
for name, df in [('Adolescent', adolescent), ('Adult', adult), ('Child', child), ('Toddler', toddler)]:
    # Get numerical columns (excluding id columns which have no predictive value)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    id_cols = [col for col in numerical_cols if 'id' in col.lower() or 'case' in col.lower()]
    numerical_cols = [col for col in numerical_cols if col not in id_cols]
    
# Impute categorical columns with mode (most frequent value)
for name, df in [('Adolescent', adolescent), ('Adult', adult), ('Child', child), ('Toddler', toddler)]:
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    missing_categorical = df[categorical_cols].isnull().sum()
    if missing_categorical.sum() > 0:
        print(f"\n  {name}:")
        for col, count in missing_categorical[missing_categorical > 0].items():
            # Use mode (most frequent value) for imputation
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            print(f"    {col}: {count} missing -> imputing with mode ('{mode_val}')")
            df[col].fillna(mode_val, inplace=True)

# ============================================================================
# STEP 6: DROP AGE_DESC COLUMN
# ============================================================================
# Reason: age_desc is redundant - we already have numerical age
# Keeping both would create multicollinearity in the model
print("\n[6/7] Dropping age_desc column...")

for name, df in [('Adolescent', adolescent), ('Adult', adult), ('Child', child), ('Toddler', toddler)]:
    if 'age_desc' in df.columns:
        df.drop('age_desc', axis=1, inplace=True)
        print(f"  OK {name}: Dropped age_desc column")

# ============================================================================
# STEP 7: DROP DATA LEAKAGE FEATURES
# ============================================================================
# CRITICAL: Remove features that would cause data leakage
# Data leakage = using information in training that wouldn't be available at prediction time
print("\n[6/7] Dropping data leakage features...")

# Features to drop:
# 1. Case_No / case_no - ID column with no predictive value (just a row identifier)
# 2. Qchat-10-Score - CRITICAL LEAKAGE: This is the sum of A1-A10 scores
#    The target variable (Class/ASD) is determined by this score (threshold >= 6)
#    Including this would give the model a direct answer, leading to 100% accuracy
#    but the model would be useless in production (we wouldn't have this score for new patients)
leakage_features = ['Case_No', 'case_no', 'Qchat-10-Score', 'qchat-10-score']

for name, df in [('Adolescent', adolescent), ('Adult', adult), ('Child', child), ('Toddler', toddler)]:
    dropped = []
    for col in leakage_features:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            dropped.append(col)
    
    if dropped:
        print(f"  OK {name}: Dropped {dropped}")
    else:
        print(f"  OK {name}: No leakage features found")

print("\n  WARNING: Qchat-10-Score removed to prevent data leakage")
print("     This feature is the sum of A1-A10 and directly reveals the target.")
print("     Models must learn from individual screening questions (A1-A10) only.")

# ============================================================================
# STEP 8: SAVE PREPROCESSED DATASETS
# ============================================================================
# Save cleaned datasets to CSV files for use in ML pipeline
# These files are now ready for training with consistent formatting and no leakage
print("\n[7/7] Saving preprocessed datasets...")

adolescent.to_csv('Autism_Adolescent_Data_Preprocessed.csv', index=False)
adult.to_csv('Autism_Adult_Data_Preprocessed.csv', index=False)
child.to_csv('Autism_Child_Data_Preprocessed.csv', index=False)
toddler.to_csv('Autism_Toddler_Data_Preprocessed.csv', index=False)

print("  OK Saved preprocessed datasets")

# ============================================================================
# PREPROCESSING SUMMARY
# ============================================================================
# Display summary statistics for each preprocessed dataset
# This helps verify that preprocessing was successful
print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)

for name, df in [('Adolescent', adolescent), ('Adult', adult), ('Child', child), ('Toddler', toddler)]:
    print(f"\n{name} Dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}")
    print(f"  Numerical columns: {df.select_dtypes(include=['int64', 'float64']).columns.tolist()}")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print("\nNext step: Run ml_pipeline.py for each dataset to train models")
