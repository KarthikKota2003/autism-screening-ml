"""
Test script to verify toddler model works with demographic-only features
"""

import requests
import json

# Test data for toddler (demographics only, A1-A10 will be ignored)
toddler_test_data = {
    'category': 'toddler',
    # A1-A10 (will be accepted from UI but NOT fed to model)
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
    # Demographics (THESE are used by the model)
    'age': '24',  # months
    'gender': 'f',
    'ethnicity': 'white european',
    'jaundice': 'no',
    'family_asd': 'no',
    'relation': 'Parent'
}

# Test data for adult (includes A1-A10)
adult_test_data = {
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

def test_prediction(url, data, test_name):
    """Test prediction endpoint"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Sending data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, data=data)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS - Prediction completed")
            # Try to extract result from HTML (basic check)
            if 'Risk' in response.text:
                print("✅ Result page rendered successfully")
            else:
                print("⚠️  Result page may have issues")
        else:
            print(f"❌ FAILED - Status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == '__main__':
    BASE_URL = 'http://127.0.0.1:5000'
    
    print("\n" + "="*60)
    print("TODDLER MODEL TESTING")
    print("="*60)
    print("\nMake sure Flask app is running on http://127.0.0.1:5000")
    print("Press Enter to start tests...")
    input()
    
    # Test 1: Toddler prediction (all NO answers, should use demographics only)
    test_prediction(
        f'{BASE_URL}/predict',
        toddler_test_data,
        'Toddler - All NO answers (demographics only)'
    )
    
    # Test 2: Adult prediction (mixed answers, uses A1-A10 + demographics)
    test_prediction(
        f'{BASE_URL}/predict',
        adult_test_data,
        'Adult - Mixed answers (A1-A10 + demographics)'
    )
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
