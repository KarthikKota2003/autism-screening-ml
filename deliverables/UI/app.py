import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename

# Add parent directory to path to import predict.py logic if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# Production-ready configuration
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)
app.config['DEBUG'] = os.environ.get('FLASK_ENV', 'development') != 'production'

# Enable CSRF protection
csrf = CSRFProtect(app)

# Configuration
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS = {
    'toddler': 'toddler_best_model.pkl',
    'child': 'child_best_model.pkl',
    'adolescent': 'adolescent_best_model.pkl',
    'adult': 'adult_best_model.pkl'
}

# Load artifacts
loaded_models = {}

def load_model_artifacts(dataset_name):
    """Load model artifacts for a specific dataset"""
    if dataset_name in loaded_models:
        return loaded_models[dataset_name]
    
    model_path = os.path.join(MODEL_DIR, MODELS[dataset_name])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading {dataset_name} model from {model_path}...")
    artifacts = joblib.load(model_path)
    loaded_models[dataset_name] = artifacts
    return artifacts

def preprocess_input(df, artifacts):
    """Preprocess input data using saved artifacts (Reused logic from predict.py)"""
    imputer = artifacts.get('imputer')
    ohe = artifacts.get('ohe')
    te = artifacts.get('te')
    scaler = artifacts['scaler']
    numerical_cols = artifacts.get('numerical_cols', [])
    ohe_cols = artifacts.get('ohe_cols', [])
    te_cols = artifacts.get('te_cols', [])
    
    # 1. Handle numerical columns
    if numerical_cols:
        for col in numerical_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        if imputer:
            X_num = pd.DataFrame(
                imputer.transform(df[numerical_cols]),
                columns=numerical_cols,
                index=df.index
            )
        else:
            X_num = df[numerical_cols].fillna(0)
    else:
        X_num = pd.DataFrame(index=df.index)
        
    # 2. Handle One-Hot Encoding
    if ohe and ohe_cols:
        for col in ohe_cols:
            if col not in df.columns:
                df[col] = 'unknown'
                
        X_ohe = pd.DataFrame(
            ohe.transform(df[ohe_cols]),
            columns=ohe.get_feature_names_out(ohe_cols),
            index=df.index
        )
    else:
        X_ohe = pd.DataFrame(index=df.index)
        
    # 3. Handle Target Encoding
    if te and te_cols:
        for col in te_cols:
            if col not in df.columns:
                df[col] = 'unknown'
                
        X_te = te.transform(df[te_cols])
    else:
        X_te = pd.DataFrame(index=df.index)
        
    # Combine features
    X_processed = pd.concat([
        X_num.reset_index(drop=True),
        X_ohe.reset_index(drop=True),
        X_te.reset_index(drop=True)
    ], axis=1)
    
    # Scale features
    X_scaled = scaler.transform(X_processed)
    
    return X_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_age', methods=['POST'])
def classify_age():
    """Classify user into category based on age"""
    data = request.json
    age_val = data.get('age')
    unit = data.get('unit', 'years')
    
    if not age_val:
        return jsonify({'error': 'Age is required'}), 400
    
    try:
        age = float(age_val)
    except ValueError:
        return jsonify({'error': 'Invalid age format'}), 400
        
    category = None
    redirect_url = None
    message = None
    
    if unit == 'months':
        if 12 <= age <= 36:
            category = 'toddler'
        elif age < 12:
            message = "Screening is typically recommended for children 12 months and older."
        else:
            # Convert to years for other categories
            age_years = age / 12
            if 4 <= age_years <= 11:
                category = 'child'
            elif 12 <= age_years <= 16:
                category = 'adolescent'
            elif age_years >= 18:
                category = 'adult'
            else:
                category = 'adolescent' # Fallback for 17
    else: # years
        if age < 1:
            # Check if they meant months or it's a baby
            message = "Screening is typically recommended for children 12 months and older."
        elif 1 <= age <= 3:
            category = 'toddler'
        elif 4 <= age <= 11:
            category = 'child'
        elif 12 <= age <= 16:
            category = 'adolescent'
        elif age >= 18:
            category = 'adult'
        else:
             # 17 years old
             category = 'adolescent'

    if category:
        redirect_url = url_for('screening', category=category, age=age, unit=unit)
        return jsonify({
            'category': category,
            'redirect_url': redirect_url,
            'message': f"Based on the age provided, the appropriate screening category is {category.capitalize()}."
        })
    else:
        return jsonify({
            'error': message or "Age out of screening range."
        })

@app.route('/screening/<category>')
def screening(category):
    if category not in MODELS:
        return redirect(url_for('index'))
    
    age = request.args.get('age', '')
    unit = request.args.get('unit', 'years')
    
    return render_template('screening.html', category=category, age=age, unit=unit)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        category = data.get('category')
        
        if category not in MODELS:
            return jsonify({'error': 'Invalid category'}), 400
            
        # Load artifacts
        artifacts = load_model_artifacts(category)
        
        # Prepare DataFrame
        # For toddler: ACCEPT A1-A10 from UI but DON'T feed to model
        # For others: Include A1-A10 in preprocessing
        if category == 'toddler':
            # Remove A1-A10 from data before creating DataFrame
            # (User can still answer them in UI, but we ignore them)
            for i in range(1, 11):
                col = f'A{i}_Score'
                if col in data:
                    del data[col]
            
            # Handle Age for toddler
            if 'age' in data:
                data['age_months'] = float(data['age'])
                del data['age']
        else:
            # For child, adolescent, adult: Convert A1-A10 to integers
            for i in range(1, 11):
                col = f'A{i}_Score'
                if col in data:
                    data[col] = int(data[col])
                else:
                    # Handle missing questions if any (though UI should enforce)
                    data[col] = 0
            
            # Handle Age for non-toddler
            if 'age' in data:
                data['age'] = float(data['age'])
                
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        X_scaled = preprocess_input(df, artifacts)
        
        # Predict
        model = artifacts['model']
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        result = {
            'prediction': 'YES' if prediction == 1 else 'NO',
            'probability': round(probability * 100, 1),
            'category': category
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        flash(f"An error occurred during analysis: {str(e)}", "error")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
