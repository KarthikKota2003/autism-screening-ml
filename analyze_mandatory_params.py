import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv('Autism_Adult_Data.csv')

print("="*80)
print("ORIGINAL DATASET ANALYSIS - MISSING VALUES")
print("="*80)

# Check for '?' which represents missing values
print("\nColumns with '?' values:")
for col in df.columns:
    if df[col].dtype == 'object':
        question_count = (df[col] == '?').sum()
        if question_count > 0:
            print(f"  {col}: {question_count} missing ({question_count/len(df)*100:.2f}%)")

# Check for NaN values
print("\nColumns with NaN values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print(f"\nTotal rows: {len(df)}")
print(f"Rows with '?' in ANY column: {(df == '?').any(axis=1).sum()}")

# Identify mandatory vs optional based on missing values
print("\n" + "="*80)
print("MANDATORY vs OPTIONAL PARAMETERS")
print("="*80)

mandatory = []
optional = []

for col in df.columns:
    if col in ['id', 'Class/ASD']:  # Skip ID and target
        continue
    
    if df[col].dtype == 'object':
        missing_count = (df[col] == '?').sum()
    else:
        missing_count = df[col].isnull().sum()
    
    if missing_count == 0:
        mandatory.append(col)
    else:
        optional.append((col, missing_count))

print("\nMANDATORY parameters (no missing values):")
for col in mandatory:
    print(f"  - {col}")

print("\nOPTIONAL parameters (have missing values):")
for col, count in optional:
    print(f"  - {col}: {count} missing ({count/len(df)*100:.2f}%)")

# Check unique values in categorical columns
print("\n" + "="*80)
print("CATEGORICAL COLUMN VALUES")
print("="*80)

for col in ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 
            'used_app_before', 'age_desc', 'relation']:
    unique_vals = df[col].unique()
    print(f"\n{col} ({len(unique_vals)} unique values):")
    print(f"  {unique_vals[:15]}")
