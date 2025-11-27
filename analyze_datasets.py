import pandas as pd
import numpy as np

# Read all datasets
print("Reading datasets...")
adolescent = pd.read_csv('Autism_Adolescent_Data.csv')
adult = pd.read_csv('Autism_Adult_Data.csv')
child = pd.read_csv('Autism_Child_Data.csv')
toddler = pd.read_csv('Autism_Toddler_Data.csv')

datasets = {
    'Adolescent': adolescent,
    'Adult': adult,
    'Child': child,
    'Toddler': toddler
}

# Analyze each dataset
for name, df in datasets.items():
    print(f"\n{'='*80}")
    print(f"=== {name.upper()} DATASET ===")
    print(f"{'='*80}")
    
    print(f"\n1. SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n2. COLUMNS:")
    print(list(df.columns))
    
    print(f"\n3. DATA TYPES:")
    print(df.dtypes)
    
    print(f"\n4. MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print(f"\n5. UNIQUE VALUES PER COLUMN:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique values")
    
    print(f"\n6. FIRST 3 ROWS:")
    print(df.head(3))
    
    print(f"\n7. CATEGORICAL vs NUMERICAL FEATURES:")
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"  Categorical ({len(categorical)}): {categorical}")
    print(f"  Numerical ({len(numerical)}): {numerical}")
    
    print(f"\n8. SAMPLE VALUES FOR CATEGORICAL FEATURES:")
    for col in categorical[:5]:  # Show first 5 categorical columns
        print(f"  {col}: {df[col].unique()[:10]}")
    
    print(f"\n9. BASIC STATISTICS FOR NUMERICAL FEATURES:")
    if len(numerical) > 0:
        print(df[numerical].describe())

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
