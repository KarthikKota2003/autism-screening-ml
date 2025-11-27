import pandas as pd
import numpy as np

# Read all datasets
print("="*80)
print("STEP 1 & 2: READING DATASETS AND COUNTING MISSING VALUES")
print("="*80)

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

# Count missing values for all datasets
print("\n" + "="*80)
print("MISSING VALUES COUNT FOR ALL DATASETS")
print("="*80)

total_missing = 0
for name, df in datasets.items():
    print(f"\n{name} Dataset:")
    print(f"  Shape: {df.shape}")
    missing_count = df.isnull().sum().sum()
    print(f"  Total Missing Values: {missing_count}")
    total_missing += missing_count
    
    if missing_count > 0:
        print(f"\n  Missing values per column:")
        missing_per_col = df.isnull().sum()
        for col, count in missing_per_col[missing_per_col > 0].items():
            pct = (count / len(df)) * 100
            print(f"    {col}: {count} ({pct:.2f}%)")
    else:
        print(f"  [OK] No missing values found!")

print(f"\n{'='*80}")
print(f"TOTAL MISSING VALUES ACROSS ALL DATASETS: {total_missing}")
print(f"{'='*80}")

# Identify target column for each dataset
print("\n" + "="*80)
print("STEP 5: CLASS BALANCE ANALYSIS")
print("="*80)

# Identify target columns (they have different names)
target_columns = {
    'Adolescent': 'Class/ASD',
    'Adult': 'Class/ASD',
    'Child': 'Class/ASD',
    'Toddler': 'Class/ASD Traits '  # Note the space at the end
}

class_balance_data = []

for name, df in datasets.items():
    target_col = target_columns[name]
    
    print(f"\n{name} Dataset - Target Column: '{target_col}'")
    print("-" * 60)
    
    # Get class distribution
    class_counts = df[target_col].value_counts()
    class_percentages = df[target_col].value_counts(normalize=True) * 100
    
    print(f"Total Samples: {len(df)}")
    print(f"\nClass Distribution:")
    
    for class_label in class_counts.index:
        count = class_counts[class_label]
        pct = class_percentages[class_label]
        print(f"  {class_label}: {count} ({pct:.2f}%)")
        
        class_balance_data.append({
            'Dataset': name,
            'Class': class_label,
            'Count': count,
            'Percentage': f"{pct:.2f}%",
            'Total Samples': len(df)
        })
    
    # Calculate imbalance ratio
    if len(class_counts) == 2:
        majority_count = class_counts.max()
        minority_count = class_counts.min()
        imbalance_ratio = majority_count / minority_count
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
        
        # Determine if balanced
        if imbalance_ratio <= 1.5:
            balance_status = "BALANCED [OK]"
        elif imbalance_ratio <= 3:
            balance_status = "MODERATELY IMBALANCED [WARNING]"
        else:
            balance_status = "HIGHLY IMBALANCED [CRITICAL]"
        
        print(f"Balance Status: {balance_status}")

# Create summary table
print("\n" + "="*80)
print("CLASS BALANCE SUMMARY TABLE")
print("="*80)

balance_df = pd.DataFrame(class_balance_data)
print(balance_df.to_string(index=False))

# Overall conclusion
print("\n" + "="*80)
print("CONCLUSION ON CLASS BALANCE")
print("="*80)

print("""
Based on the analysis above:

1. ADOLESCENT Dataset: 
   - Needs to be evaluated based on the imbalance ratio shown above
   
2. ADULT Dataset:
   - Needs to be evaluated based on the imbalance ratio shown above
   
3. CHILD Dataset:
   - Needs to be evaluated based on the imbalance ratio shown above
   
4. TODDLER Dataset:
   - Needs to be evaluated based on the imbalance ratio shown above

A dataset is considered:
- BALANCED: Imbalance ratio <= 1.5:1
- MODERATELY IMBALANCED: Imbalance ratio between 1.5:1 and 3:1
- HIGHLY IMBALANCED: Imbalance ratio > 3:1
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - WAITING FOR USER CONFIRMATION ON IMPUTATION")
print("="*80)
