import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('Autism_Adolescent_Data_Preprocessed.csv')

# Calculate sum of scores
score_cols = [f'A{i}_Score' for i in range(1, 11)]
df['score_sum'] = df[score_cols].sum(axis=1)

# Check relationship with target
target_col = 'Class/ASD'
print("Score Sum vs Target:")
print(df.groupby(target_col)['score_sum'].describe())

# Check if there is a perfect threshold
# Usually threshold is 7 or similar
print("\nPotential Thresholds:")
for threshold in range(11):
    pred = (df['score_sum'] >= threshold).map({True: 'YES', False: 'NO'})
    accuracy = (pred == df[target_col]).mean()
    print(f"Threshold >= {threshold}: Accuracy = {accuracy:.4f}")
