"""
Unit and Integration Tests for Autism Screening Flask Application

Tests:
1. Route availability
2. Age classification logic
3. Form submission with valid/invalid data
4. Model loading and prediction pipeline integration
5. Security (CSRF, XSS, Input validation)
"""

import pytest
import json
from app import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    with app.test_client() as client:
        yield client

# ============================================================================
# Route Availability Tests
# ============================================================================

def test_index_route(client):
    """Test home page loads successfully"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'NeuroScreen AI' in response.data or b'Early Detection' in response.data

def test_screening_route_toddler(client):
    """Test toddler screening page loads"""
    response = client.get('/screening/toddler')
    assert response.status_code == 200
    assert b'Toddler' in response.data or b'toddler' in response.data

def test_screening_route_child(client):
    """Test child screening page loads"""
    response = client.get('/screening/child')
    assert response.status_code == 200
    assert b'Child' in response.data or b'child' in response.data

def test_screening_route_adolescent(client):
    """Test adolescent screening page loads"""
    response = client.get('/screening/adolescent')
    assert response.status_code == 200
    assert b'Adolescent' in response.data or b'adolescent' in response.data

def test_screening_route_adult(client):
    """Test adult screening page loads"""
    response = client.get('/screening/adult')
    assert response.status_code == 200
    assert b'Adult' in response.data or b'adult' in response.data

def test_invalid_category_redirect(client):
    """Test invalid category redirects to home"""
    response = client.get('/screening/invalid')
    assert response.status_code == 302  # Redirect

# ============================================================================
# Age Classification Tests
# ============================================================================

def test_classify_age_toddler_months(client):
    """Test age classification for toddler (months)"""
    response = client.post('/classify_age',
                          data=json.dumps({'age': 24, 'unit': 'months'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['category'] == 'toddler'

def test_classify_age_child_years(client):
    """Test age classification for child (years)"""
    response = client.post('/classify_age',
                          data=json.dumps({'age': 8, 'unit': 'years'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['category'] == 'child'

def test_classify_age_adolescent(client):
    """Test age classification for adolescent"""
    response = client.post('/classify_age',
                          data=json.dumps({'age': 14, 'unit': 'years'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['category'] == 'adolescent'

def test_classify_age_adult(client):
    """Test age classification for adult"""
    response = client.post('/classify_age',
                          data=json.dumps({'age': 25, 'unit': 'years'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['category'] == 'adult'

def test_classify_age_edge_case_17(client):
    """Test age classification for 17 years (edge case)"""
    response = client.post('/classify_age',
                          data=json.dumps({'age': 17, 'unit': 'years'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['category'] == 'adolescent'

def test_classify_age_missing_age(client):
    """Test age classification with missing age"""
    response = client.post('/classify_age',
                          data=json.dumps({'unit': 'years'}),
                          content_type='application/json')
    assert response.status_code == 400

def test_classify_age_invalid_format(client):
    """Test age classification with invalid age format"""
    response = client.post('/classify_age',
                          data=json.dumps({'age': 'invalid', 'unit': 'years'}),
                          content_type='application/json')
    assert response.status_code == 400

# ============================================================================
# Form Submission Tests
# ============================================================================

def test_predict_missing_category(client):
    """Test prediction with missing category"""
    response = client.post('/predict', data={})
    assert response.status_code == 400

def test_predict_invalid_category(client):
    """Test prediction with invalid category"""
    response = client.post('/predict', data={'category': 'invalid'})
    assert response.status_code == 400

def test_predict_valid_adult_data(client):
    """Test prediction with valid adult data"""
    data = {
        'category': 'adult',
        'A1_Score': '1',
        'A2_Score': '0',
        'A3_Score': '1',
        'A4_Score': '0',
        'A5_Score': '1',
        'A6_Score': '0',
        'A7_Score': '1',
        'A8_Score': '0',
        'A9_Score': '1',
        'A10_Score': '0',
        'age': '25',
        'gender': 'm',
        'ethnicity': 'white european',
        'jaundice': 'no',
        'family_asd': 'no',
        'contry_of_res': 'United States',
        'used_app_before': 'no',
        'relation': 'Self'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Result' in response.data or b'result' in response.data

# ============================================================================
# Security Tests
# ============================================================================

def test_xss_prevention_in_age_input(client):
    """Test XSS prevention in age input"""
    response = client.post('/classify_age',
                          data=json.dumps({'age': '<script>alert("XSS")</script>', 'unit': 'years'}),
                          content_type='application/json')
    assert response.status_code == 400  # Should reject invalid input

def test_sql_injection_prevention(client):
    """Test SQL injection prevention (though we don't use SQL)"""
    data = {
        'category': 'adult',
        'A1_Score': '1; DROP TABLE users;--',
        'age': '25',
        'gender': 'm',
        'ethnicity': 'white european',
        'jaundice': 'no',
        'family_asd': 'no',
        'contry_of_res': 'United States',
        'used_app_before': 'no',
        'relation': 'Self'
    }
    # Should handle gracefully without crashing
    response = client.post('/predict', data=data)
    # Either 400 (validation error) or 302 (redirect on error) is acceptable
    assert response.status_code in [200, 302, 400]

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
