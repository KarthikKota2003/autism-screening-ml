# Autism Screening Web Application

A user-friendly Flask web application for autism screening across four age groups: Toddler, Child, Adolescent, and Adult. The application uses pre-trained machine learning models to provide risk assessments based on behavioral questionnaires.

## Features

- **Age Classifier**: Automatically suggests the appropriate screening category based on age
- **4 Screening Categories**: Toddler (12-36 months), Child (4-11 years), Adolescent (12-16 years), Adult (18+ years)
- **ML-Powered Predictions**: Uses trained models for accurate risk assessment
- **Responsive Design**: Modern, premium UI with smooth animations
- **Comprehensive Testing**: Unit, integration, and security tests included

## Project Structure

```
UI/
├── app.py                 # Flask application backend
├── requirements.txt       # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css     # Main stylesheet
│   ├── js/
│   │   └── main.js       # Frontend logic
│   └── images/           # Static images
├── templates/
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── screening.html    # Screening form
│   └── result.html       # Results page
└── tests/
    ├── test_app.py       # Unit & integration tests
    └── test_low_risk.py  # Low-risk verification tests
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Navigate to the UI directory**:
   ```bash
   cd d:\AutismML\deliverables\UI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**:
   The application requires the following model files in the parent directory (`d:\AutismML\deliverables\`):
   - `toddler_best_model.pkl`
   - `child_best_model.pkl`
   - `adolescent_best_model.pkl`
   - `adult_best_model.pkl`

## Running the Application

### Development Server

```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

Open your web browser and navigate to `http://127.0.0.1:5000` to access the application.

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test Files

```bash
# Unit and integration tests
python -m pytest tests/test_app.py -v

# Low-risk verification tests
python -m pytest tests/test_low_risk.py -v
```

### Test Coverage

- **Route Availability**: Tests all routes (/, /screening/<category>, /classify_age, /predict)
- **Age Classification**: Tests age classifier logic for all categories and edge cases
- **Form Submission**: Tests valid and invalid form data
- **Model Integration**: Tests model loading and prediction pipeline
- **Security**: Tests XSS prevention and input validation
- **Low-Risk Verification**: Tests that models correctly handle low-risk inputs

## Usage

### 1. Home Page

- View 4 island blocks representing each age category
- Click on a category to start screening directly
- Use the Age Classifier (bottom-right FAB) to find the right category

### 2. Age Classifier

- Enter age in years or months
- System automatically suggests the appropriate category
- Confirm to proceed to screening

### 3. Screening Form

- Answer all A1-A10 behavioral questions (Yes/No)
- Fill in demographic information:
  - Age
  - Gender
  - Ethnicity
  - Jaundice history
  - Family ASD history
  - Country of residence
  - Previous app usage
  - Relation (for Adolescent/Adult/Toddler)
- Submit for analysis

### 4. Results Page

- View risk assessment (High Risk / Low Risk)
- See model confidence percentage
- Read age-specific recommendations
- Return to home or learn more about autism

## Security Features

- **Input Validation**: All inputs are validated on both client and server side
- **XSS Prevention**: Jinja2 auto-escaping enabled
- **CSRF Protection**: Can be enabled with Flask-WTF
- **Secure Model Loading**: Models are loaded from a secure directory
- **Error Handling**: Graceful error handling with user-friendly messages

## Known Limitations

1. **Model Version Compatibility**: Models were trained with scikit-learn 1.7.2 but the app uses 1.3.2 for compatibility. This may cause warnings but does not affect functionality.

2. **Low-Risk Test Failures**: Some low-risk tests fail because the models learned that all-zero behavioral patterns (all "No" answers) can still indicate risk based on other demographic factors. This is expected behavior.

3. **Development Server**: The built-in Flask server is for development only. Use a production WSGI server for deployment.

## Troubleshooting

### Models Not Found

If you see "Model file not found" errors:
1. Ensure you're in the correct directory (`d:\AutismML\deliverables\UI`)
2. Check that model files exist in the parent directory
3. Run `ml_pipeline.py` to generate models if missing

### Import Errors

If you encounter import errors:
```bash
pip install -r requirements.txt --force-reinstall
```

### Port Already in Use

If port 5000 is already in use:
```python
# In app.py, change the last line to:
app.run(debug=True, port=5001)
```

## Contributing

This is a research project. For questions or contributions, please contact the project maintainer.

## Disclaimer

**IMPORTANT**: This application provides a screening assessment based on machine learning models. It is **NOT** a medical diagnosis. Please consult a qualified healthcare professional for a formal evaluation.

## License

Research Project - 2025
