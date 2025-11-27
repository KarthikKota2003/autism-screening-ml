# Q-CHAT-10 Screening Questions (A1-A10) Explained

## Overview

The **Q-CHAT-10** (Quantitative Checklist for Autism in Toddlers) is a 10-item parent-reported screening questionnaire used to identify early signs of Autism Spectrum Disorder (ASD) in children, typically between 18-24 months of age.

Each question (A1-A10) is scored on a **5-point Likert scale**, with responses coded as binary values (0 or 1) in this dataset.

---

## The 10 Screening Questions

### A1: Response to Name
**Question:** *Does your child look at you when you call his/her name?*

**What it assesses:** 
- Response to social cues
- Attentiveness to their name
- Basic social communication skills

**Why it matters:** Children with ASD may not consistently respond when their name is called, indicating reduced social awareness.

---

### A2: Eye Contact
**Question:** *How easy is it for you to get eye contact with your child?*

**What it assesses:**
- Use of eye contact during interactions
- Social engagement
- Joint attention abilities

**Why it matters:** Reduced or atypical eye contact is a common early indicator of autism.

---

### A3: Instrumental Pointing (Requesting)
**Question:** *Does your child point to indicate that s/he wants something? (e.g., a toy that is out of reach)*

**What it assesses:**
- Instrumental pointing (pointing to request)
- Non-verbal communication for needs
- Understanding of cause and effect

**Why it matters:** This tests whether the child uses pointing as a tool to communicate their desires.

---

### A4: Declarative Pointing (Sharing Interest)
**Question:** *Does your child point to share interest with you? (e.g., pointing at an interesting sight)*

**What it assesses:**
- Declarative pointing (pointing to share)
- Joint attention
- Social sharing of experiences

**Why it matters:** This is a more advanced social skill than A3. Children with ASD may point to request but not to share experiences.

---

### A5: Pretend Play
**Question:** *Does your child pretend? (e.g., care for dolls, talk on a toy phone)*

**What it assesses:**
- Symbolic/pretend play abilities
- Imagination and creativity
- Social-cognitive development

**Why it matters:** Reduced or absent pretend play is a hallmark feature of autism in early childhood.

---

### A6: Gaze Following
**Question:** *Does your child follow where you're looking?*

**What it assesses:**
- Gaze following ability
- Joint attention
- Understanding of social cues

**Why it matters:** Following another person's gaze is essential for shared attention and social learning.

---

### A7: Empathy and Comfort
**Question:** *If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g., stroking hair, hugging them)*

**What it assesses:**
- Empathy
- Emotional reciprocity
- Social-emotional awareness

**Why it matters:** Children with ASD may have difficulty recognizing or responding to others' emotional states.

---

### A8: Early Language Development
**Question:** *Would you describe your child's first words as:*
- Very typical
- Quite typical
- Slightly unusual
- Very unusual
- My child doesn't speak

**What it assesses:**
- Typicality of early language development
- Communication milestones
- Verbal development patterns

**Why it matters:** Atypical or delayed language development can be an early sign of autism.

---

### A9: Simple Gestures
**Question:** *Does your child use simple gestures? (e.g., wave goodbye)*

**What it assesses:**
- Non-verbal communication skills
- Social gestures
- Imitation abilities

**Why it matters:** Reduced use of conventional gestures is common in children with ASD.

---

### A10: Unusual Fixations
**Question:** *Does your child stare at nothing with no apparent purpose?*

**What it assesses:**
- Repetitive behaviors
- Unusual fixations or interests
- Atypical sensory processing

**Why it matters:** Staring at objects or into space can indicate restricted or repetitive behaviors associated with autism.

---

## Scoring and Interpretation

### In This Dataset
- Each question is coded as **binary (0 or 1)**
- **0** typically indicates typical development
- **1** typically indicates atypical development or ASD traits

### Clinical Interpretation
- **Total Score = Sum of A1-A10** (range: 0-10)
- **Threshold:** A score of **≥3** suggests the need for further multi-disciplinary assessment
- **Higher scores** indicate more autistic traits

### Important Notes

> [!CAUTION]
> The Q-CHAT-10 is a **screening tool**, not a diagnostic instrument. A high score indicates the need for professional evaluation, not a definitive diagnosis.

> [!IMPORTANT]
> In the toddler dataset, the target variable `Class/ASD` is **deterministically derived** from the sum of A1-A10:
> - If (A1 + A2 + ... + A10) > 3, then Class = YES (ASD)
> - Otherwise, Class = NO

This explains why machine learning models achieve near-perfect accuracy on this dataset—they are learning the underlying screening rule, not discovering new diagnostic patterns.

---

## Clinical Domains Assessed

The Q-CHAT-10 evaluates three key domains:

1. **Joint Attention** (A2, A4, A6)
   - Eye contact
   - Declarative pointing
   - Gaze following

2. **Social Communication** (A1, A3, A7, A9)
   - Response to name
   - Instrumental pointing
   - Empathy
   - Gestures

3. **Pretend Play & Behavior** (A5, A8, A10)
   - Pretend play
   - Language development
   - Repetitive behaviors

---

## References

- Allison, C., et al. (2008). The Q-CHAT (Quantitative CHecklist for Autism in Toddlers): A normally distributed quantitative measure of autistic traits at 18-24 months of age.
- Autism Research Centre, University of Cambridge
- Psychology Tools: Q-CHAT-10 Screening Questionnaire

---

## Usage in Machine Learning

When using A1-A10 as input features for ML models:

✅ **Do:**
- Use individual A1-A10 scores as separate features
- Apply appropriate scaling (they're already 0-1 binary)
- Consider feature interactions (e.g., A3 vs A4 pointing differences)

❌ **Don't:**
- Include the sum (Qchat-10-Score) as a feature—this causes data leakage
- Include the Case_No or ID columns
- Assume perfect accuracy means a good model—check if the target is deterministic
