# Categorical Columns Analysis - All Datasets

## Quick Reference: Categorical Columns by Dataset

### Adolescent Dataset (9 categorical columns)
1. `gender`
2. `ethnicity`
3. `jundice`
4. `austim`
5. `contry_of_res`
6. `used_app_before`
7. `age_desc`
8. `relation`
9. `Class/ASD` (TARGET)

### Adult Dataset (10 categorical columns)
1. `age` (stored as string)
2. `gender`
3. `ethnicity`
4. `jundice`
5. `austim`
6. `contry_of_res`
7. `used_app_before`
8. `age_desc`
9. `relation`
10. `Class/ASD` (TARGET)

### Child Dataset (10 categorical columns)
1. `age` (stored as string)
2. `gender`
3. `ethnicity`
4. `jundice`
5. `austim`
6. `contry_of_res`
7. `used_app_before`
8. `age_desc`
9. `relation`
10. `Class/ASD` (TARGET)

### Toddler Dataset (6 categorical columns)
1. `Sex`
2. `Ethnicity`
3. `Jaundice`
4. `Family_mem_with_ASD`
5. `Who completed the test`
6. `Class/ASD Traits ` (TARGET - note the trailing space)

---

## Detailed Values for Each Categorical Column

### 1. Gender/Sex (Binary)
**Column Names:** `gender` (Adolescent, Adult, Child), `Sex` (Toddler)

| Dataset | Values | Distribution |
|---------|--------|--------------|
| Adolescent | 'm', 'f' | m: 67.31%, f: 32.69% |
| Adult | 'f', 'm' | f: 50.14%, m: 49.86% |
| Child | 'm', 'f' | m: 66.78%, f: 33.22% |
| Toddler | 'm', 'f' | m: 69.73%, f: 30.27% |

**Unique Values:** 2
**Recommendation:** Binary encoding or One-Hot Encoding

---

### 2. Ethnicity (Multi-class)
**Column Names:** `ethnicity` (Adolescent, Adult, Child), `Ethnicity` (Toddler)

#### Adolescent Dataset (9 unique values):
- 'Hispanic', 'Black', '?', 'White-European', 'Middle Eastern ', 'Asian', 'South Asian', 'Others', 'Latino'

#### Adult Dataset (12 unique values):
- 'White-European', 'Latino', '?', 'Others', 'Black', 'Asian', 'Middle Eastern ', 'Pasifika', 'South Asian', 'Hispanic', 'Turkish', 'others'

#### Child Dataset (11 unique values):
- 'Others', 'Middle Eastern ', '?', 'White-European', 'Black', 'South Asian', 'Asian', 'Pasifika', 'Hispanic', 'Turkish', 'Latino'

#### Toddler Dataset (11 unique values):
- 'middle eastern', 'White European', 'Hispanic', 'black', 'asian', 'south asian', 'Native Indian', 'Others', 'Latino', 'mixed', 'Pacifica'

**Unique Values:** 9-12 (varies by dataset)
**Note:** Inconsistent capitalization across datasets
**Recommendation:** One-Hot Encoding required

---

### 3. Jaundice/Jundice (Binary)
**Column Names:** `jundice` (Adolescent, Adult, Child), `Jaundice` (Toddler)

| Dataset | Values | Distribution |
|---------|--------|--------------|
| Adolescent | 'yes', 'no' | yes: 11.54%, no: 88.46% |
| Adult | 'no', 'yes' | no: 89.49%, yes: 10.51% |
| Child | 'no', 'yes' | no: 89.04%, yes: 10.96% |
| Toddler | 'no', 'yes' | no: 72.68%, yes: 27.32% |

**Unique Values:** 2
**Note:** Spelling inconsistency - "jundice" vs "Jaundice"
**Recommendation:** Binary encoding or One-Hot Encoding

---

### 4. Autism Family History (Binary)
**Column Names:** `austim` (Adolescent, Adult, Child), `Family_mem_with_ASD` (Toddler)

| Dataset | Values | Distribution |
|---------|--------|--------------|
| Adolescent | 'yes', 'no' | yes: 18.27%, no: 81.73% |
| Adult | 'no', 'yes' | no: 82.95%, yes: 17.05% |
| Child | 'no', 'yes' | no: 83.56%, yes: 16.44% |
| Toddler | 'no', 'yes' | no: 83.87%, yes: 16.13% |

**Unique Values:** 2
**Recommendation:** Binary encoding or One-Hot Encoding

---

### 5. Country of Residence (High Cardinality)
**Column Names:** `contry_of_res` (Adolescent, Adult, Child), NOT PRESENT in Toddler

| Dataset | Unique Values | Top 5 Countries |
|---------|---------------|-----------------|
| Adolescent | 33 | Austria, AmericanSamoa, United Kingdom, Albania, Belgium |
| Adult | 67 | United States, Brazil, Spain, Egypt, New Zealand |
| Child | 52 | Jordan, United States, Egypt, United Kingdom, Bahrain |

**Note:** High cardinality feature (33-67 unique values)
**Recommendation:** Consider dropping, grouping, or using target encoding instead of One-Hot

---

### 6. Used App Before (Binary)
**Column Names:** `used_app_before` (Adolescent, Adult, Child), NOT PRESENT in Toddler

| Dataset | Values | Distribution |
|---------|--------|--------------|
| Adolescent | 'no', 'yes' | no: 100.00%, yes: 0.00% |
| Adult | 'no', 'yes' | no: 99.72%, yes: 0.28% |
| Child | 'no', 'yes' | no: 99.66%, yes: 0.34% |

**Unique Values:** 2
**Note:** Extremely imbalanced - almost all "no"
**Recommendation:** Consider dropping (low variance) or binary encoding

---

### 7. Age Description (Low Variance)
**Column Names:** `age_desc` (Adolescent, Adult, Child), NOT PRESENT in Toddler

| Dataset | Values | Distribution |
|---------|--------|--------------|
| Adolescent | '12-16 years', '12-15 years' | 12-16 years: 98.08%, 12-15 years: 1.92% |
| Adult | '18 and more' | 18 and more: 100.00% |
| Child | '4-11 years' | 4-11 years: 100.00% |

**Unique Values:** 1-2
**Note:** Low/no variance
**Recommendation:** Consider dropping (redundant with dataset type)

---

### 8. Relation/Who Completed Test (Multi-class)
**Column Names:** `relation` (Adolescent, Adult, Child), `Who completed the test` (Toddler)

#### Adolescent Dataset (6 values):
- 'Parent': 94.23%, 'Relative': 2.88%, '?': 0.96%, 'Self': 0.96%, 'Health care professional': 0.96%, 'Others': 0.00%

#### Adult Dataset (6 values):
- 'Self': 88.78%, 'Parent': 6.11%, '?': 2.56%, 'Health care professional': 1.42%, 'Relative': 1.14%, 'Others': 0.00%

#### Child Dataset (6 values):
- 'Parent': 95.21%, '?': 2.40%, 'Self': 1.37%, 'Relative': 0.68%, 'Health care professional': 0.34%, 'Others': 0.00%

#### Toddler Dataset (5 values):
- 'family member': 96.58%, 'Health Care Professional': 2.28%, 'Health care professional': 0.47%, 'Self': 0.38%, 'Others': 0.28%

**Unique Values:** 5-6
**Note:** Inconsistent capitalization in Toddler dataset
**Recommendation:** One-Hot Encoding

---

### 9. Age (Categorical in Adult/Child)
**Column Names:** `age` (Adult, Child only)

#### Adult Dataset (47 unique values):
- Range: Various ages as strings (e.g., '26', '24', '27', '35', '40', etc.)
- Contains '?' for missing/unknown

#### Child Dataset (9 unique values):
- Values: '6', '5', '4', '11', '10', '8', '7', '9', '?'

**Note:** Stored as categorical but represents numerical age
**Recommendation:** Convert to numerical, handle '?' as missing value

---

### 10. Target Variable (Binary)
**Column Names:** `Class/ASD` (Adolescent, Adult, Child), `Class/ASD Traits ` (Toddler)

| Dataset | Values | Distribution |
|---------|--------|--------------|
| Adolescent | 'NO', 'YES' | NO: 39.42%, YES: 60.58% |
| Adult | 'NO', 'YES' | NO: 73.15%, YES: 26.85% |
| Child | 'NO', 'YES' | NO: 51.71%, YES: 48.29% |
| Toddler | 'No', 'Yes' | No: 30.93%, Yes: 69.07% |

**Unique Values:** 2
**Note:** Inconsistent capitalization (NO/YES vs No/Yes)
**Recommendation:** DO NOT encode - this is the target variable

---

## Summary Table: Encoding Recommendations

| Column | Type | Unique Values | Recommendation | Priority |
|--------|------|---------------|----------------|----------|
| gender/Sex | Binary | 2 | One-Hot or Binary | HIGH |
| ethnicity/Ethnicity | Multi-class | 9-12 | One-Hot | HIGH |
| jundice/Jaundice | Binary | 2 | One-Hot or Binary | HIGH |
| austim/Family_mem_with_ASD | Binary | 2 | One-Hot or Binary | HIGH |
| contry_of_res | High Cardinality | 33-67 | Drop or Target Encode | LOW |
| used_app_before | Binary (Low Var) | 2 | Drop or Binary | LOW |
| age_desc | Low Variance | 1-2 | Drop | LOW |
| relation/Who completed | Multi-class | 5-6 | One-Hot | MEDIUM |
| age (Adult/Child) | Numerical | 9-47 | Convert to Numeric | HIGH |
| Class/ASD | Target | 2 | Label Encode (0/1) | N/A |

---

## Data Quality Issues to Address

1. **Inconsistent Capitalization:**
   - Ethnicity values: 'White-European' vs 'White European', 'asian' vs 'Asian'
   - Target variable: 'YES/NO' vs 'Yes/No'
   - Relation: 'Health care professional' vs 'Health Care Professional'

2. **Missing Value Indicators:**
   - '?' used in age, ethnicity, relation columns

3. **Spelling Inconsistency:**
   - 'jundice' vs 'Jaundice' (correct spelling)

4. **Column Name Inconsistency:**
   - Different names for same concept across datasets
   - Trailing space in 'Class/ASD Traits '

5. **Low Variance Features:**
   - `used_app_before`: 99%+ are "no"
   - `age_desc`: Single value in Adult/Child datasets

---

## Recommended Preprocessing Steps

1. **Standardize capitalization** across all categorical values
2. **Handle '?' as missing values** (though currently none exist)
3. **Convert age to numerical** in Adult and Child datasets
4. **Drop low-variance features:** `age_desc`, `used_app_before`
5. **Consider dropping or encoding differently:** `contry_of_res` (high cardinality)
6. **Apply One-Hot Encoding to:** gender, ethnicity, jaundice, autism history, relation
7. **Label encode target variable:** Class/ASD → 0/1

---

**AWAITING USER CONFIRMATION:** Please specify which columns you would like to perform One-Hot Encoding on.
