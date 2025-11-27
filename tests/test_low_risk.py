"""
Low-Risk Verification Tests

Tests that models correctly predict "No" / Low Risk for inputs with known
low autistic traits across all 4 categories (Toddler, Child, Adolescent, Adult).

This ensures the models are not biased towards predicting high risk.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client

# ============================================================================
# Low-Risk Test Cases
# ============================================================================

def test_low_risk_toddler(client):
    """Test low-risk prediction for toddler"""
    data = {
        'category': 'toddler',
        # All A1-A10 answered "No" (0)
        'A1_Score': '0',
        'A2_Score': '0',
        'A3_Score': '0',
        'A4_Score': '0',
        'A5_Score': '0',
        'A6_Score': '0',
        'A7_Score': '0',
        'A8_Score': '0',
        'A9_Score': '0',
        'A10_Score': '0',
        'age': '24',  # 24 months
        'gender': 'f',
        'ethnicity': 'white european',
        'jaundice': 'no',
        'family_asd': 'no',
        'relation': 'Parent'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    # Check that result indicates low risk
    assert b'Low Risk' in response.data or b'NO' in response.data

def test_low_risk_child(client):
    """Test low-risk prediction for child"""
    data = {
        'category': 'child',
        # All A1-A10 answered "No" (0)
        'A1_Score': '0',
        'A2_Score': '0',
        'A3_Score': '0',
        'A4_Score': '0',
        'A5_Score': '0',
        'A6_Score': '0',
        'A7_Score': '0',
        'A8_Score': '0',
        'A9_Score': '0',
        'A10_Score': '0',
        'age': '7',
        'gender': 'm',
        'ethnicity': 'asian',
        'jaundice': 'no',
        'family_asd': 'no',
        'contry_of_res': 'United States',
        'used_app_before': 'no'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Low Risk' in response.data or b'NO' in response.data

def test_low_risk_adolescent(client):
    """Test low-risk prediction for adolescent"""
    data = {
        'category': 'adolescent',
        # All A1-A10 answered "No" (0)
        'A1_Score': '0',
        'A2_Score': '0',
        'A3_Score': '0',
        'A4_Score': '0',
        'A5_Score': '0',
        'A6_Score': '0',
        'A7_Score': '0',
        'A8_Score': '0',
        'A9_Score': '0',
        'A10_Score': '0',
        'age': '14',
        'gender': 'f',
        'ethnicity': 'latino',
        'jaundice': 'no',
        'family_asd': 'no',
        'contry_of_res': 'Canada',
        'used_app_before': 'no',
        'relation': 'Self'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Low Risk' in response.data or b'NO' in response.data

def test_low_risk_adult(client):
    """Test low-risk prediction for adult"""
    data = {
        'category': 'adult',
        # All A1-A10 answered "No" (0)
        'A1_Score': '0',
        'A2_Score': '0',
        'A3_Score': '0',
        'A4_Score': '0',
        'A5_Score': '0',
        'A6_Score': '0',
        'A7_Score': '0',
        'A8_Score': '0',
        'A9_Score': '0',
        'A10_Score': '0',
        'age': '30',
        'gender': 'm',
        'ethnicity': 'black',
        'jaundice': 'no',
        'family_asd': 'no',
        'contry_of_res': 'United Kingdom',
        'used_app_before': 'no',
        'relation': 'Self'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Low Risk' in response.data or b'NO' in response.data

# ============================================================================
# Mixed Risk Test Cases (Some Yes, Some No)
# ============================================================================

def test_mixed_risk_adult(client):
    """Test mixed-risk prediction for adult (3 Yes, 7 No)"""
    data = {
        'category': 'adult',
        'A1_Score': '1',
        'A2_Score': '0',
        'A3_Score': '1',
        'A4_Score': '0',
        'A5_Score': '0',
        'A6_Score': '0',
        'A7_Score': '1',
        'A8_Score': '0',
        'A9_Score': '0',
        'A10_Score': '0',
        'age': '28',
        'gender': 'f',
        'ethnicity': 'south asian',
        'jaundice': 'no',
        'family_asd': 'no',
        'contry_of_res': 'India',
        'used_app_before': 'no',
        'relation': 'Self'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    # Should return a result (either low or high risk)
    assert b'Risk' in response.data

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
