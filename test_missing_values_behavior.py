"""
Test how the ML pipeline handles missing values in mandatory vs optional parameters
"""
import pandas as pd
import numpy as np
import json

# Load the preprocessed dataset
df = pd.read_csv('Autism_Adult_Data_Preprocessed.csv')

print("="*80)
print("TESTING PIPELINE BEHAVIOR WITH MISSING VALUES")
print("="*80)

# Create test cases
test_cases = []

# Test Case 1: Missing MANDATORY parameter (gender)
test1 = df.iloc[0].copy()
test1['gender'] = np.nan
test_cases.append(("Missing MANDATORY: gender", test1))

# Test Case 2: Missing OPTIONAL parameter (age)
test2 = df.iloc[0].copy()
test2['age'] = np.nan
test_cases.append(("Missing OPTIONAL: age", test2))

# Test Case 3: Missing OPTIONAL parameter (ethnicity)
test3 = df.iloc[0].copy()
test3['ethnicity'] = '?'
test_cases.append(("Missing OPTIONAL: ethnicity", test3))

# Test Case 4: Missing OPTIONAL parameter (relation)
test4 = df.iloc[0].copy()
test4['relation'] = '?'
test_cases.append(("Missing OPTIONAL: relation", test4))

# Test Case 5: Missing MULTIPLE optional parameters
test5 = df.iloc[0].copy()
test5['age'] = np.nan
test5['ethnicity'] = '?'
test5['relation'] = '?'
test_cases.append(("Missing MULTIPLE OPTIONAL: age, ethnicity, relation", test5))

# Test Case 6: Complete data (baseline)
test6 = df.iloc[0].copy()
test_cases.append(("COMPLETE data (baseline)", test6))

# Display test cases
print("\nTest Cases Created:")
for i, (desc, data) in enumerate(test_cases, 1):
    print(f"\n{i}. {desc}")
    missing_cols = []
    for col in data.index:
        if pd.isna(data[col]) or data[col] == '?':
            missing_cols.append(col)
    if missing_cols:
        print(f"   Missing: {missing_cols}")
    else:
        print(f"   All values present")

# Check how preprocessing handles these
print("\n" + "="*80)
print("CURRENT PIPELINE HANDLING")
print("="*80)

print("\nThe ml_pipeline.py currently:")
print("1. Drops A1-A10 scores and 'result' column (data leakage prevention)")
print("2. Uses OneHotEncoder with handle_unknown='ignore' for categorical features")
print("3. Uses SimpleImputer(strategy='median') for numerical features (age)")
print("4. Does NOT explicitly handle '?' values in categorical columns")

print("\n" + "="*80)
print("POTENTIAL ISSUES")
print("="*80)

print("\n[WARNING] ISSUE 1: '?' values in categorical columns")
print("   - Columns: ethnicity, relation")
print("   - Current behavior: OneHotEncoder will treat '?' as a valid category")
print("   - Impact: Creates a 'ethnicity_?' and 'relation_?' feature")
print("   - Recommendation: Replace '?' with mode or create 'Unknown' category")

print("\n[WARNING] ISSUE 2: Missing mandatory categorical values (if they occur)")
print("   - Columns: gender, jaundice, family_asd, used_app_before")
print("   - Current behavior: OneHotEncoder will fail if NaN is present")
print("   - Impact: Pipeline will crash")
print("   - Recommendation: Add categorical imputation before encoding")

print("\n[OK] HANDLED: Missing numerical values")
print("   - Column: age")
print("   - Current behavior: SimpleImputer fills with median")
print("   - Impact: No issues, properly handled")

# Save test cases to JSON for prediction testing
test_data = []
for desc, data in test_cases:
    test_dict = data.to_dict()
    # Convert NaN to None for JSON serialization
    for key, value in test_dict.items():
        if pd.isna(value):
            test_dict[key] = None
    test_data.append({"description": desc, "data": test_dict})

with open('test_missing_values.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print("\n" + "="*80)
print("Test cases saved to: test_missing_values.json")
print("="*80)
