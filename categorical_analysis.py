import pandas as pd
import numpy as np

# Read all datasets
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

print("="*80)
print("CATEGORICAL COLUMNS AND THEIR VALUES - DETAILED ANALYSIS")
print("="*80)

for name, df in datasets.items():
    print(f"\n{'='*80}")
    print(f"{name.upper()} DATASET")
    print(f"{'='*80}")
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nTotal Categorical Columns: {len(categorical_cols)}")
    print(f"Columns: {categorical_cols}")
    
    print(f"\n{'-'*80}")
    print("DETAILED VALUES FOR EACH CATEGORICAL COLUMN")
    print(f"{'-'*80}")
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  - Unique Values Count: {df[col].nunique()}")
        print(f"  - Data Type: {df[col].dtype}")
        
        # Get value counts
        value_counts = df[col].value_counts()
        
        print(f"  - Value Distribution:")
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"      '{value}': {count} ({percentage:.2f}%)")
        
        # Show all unique values
        unique_vals = sorted(df[col].unique().astype(str))
        print(f"  - All Unique Values: {unique_vals}")
        
        # Check for missing values
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"  - Missing Values: {missing}")
        
        print(f"  {'-'*76}")

print("\n" + "="*80)
print("SUMMARY TABLE: CATEGORICAL COLUMNS ACROSS ALL DATASETS")
print("="*80)

# Create a summary
summary_data = []
for name, df in datasets.items():
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        summary_data.append({
            'Dataset': name,
            'Column': col,
            'Unique Values': df[col].nunique(),
            'Sample Values': ', '.join([str(x) for x in df[col].unique()[:5]])
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE - WAITING FOR USER INPUT")
print("="*80)
