import json
import requests
import os

# Check if label_mapping.json exists in the correct directory
label_file = "deployment/pubmedbert/label_mapping.json"
print(f"Checking for label mapping at: {label_file}")
print(f"File exists: {os.path.exists(label_file)}")

if not os.path.exists(label_file):
    print("Error: label_mapping.json not found. Please check the deployment directory.")
    exit(1)

# Load label mapping
try:
    with open(label_file, "r") as f:
        label_mapping = json.load(f)
    print(f"Label mapping loaded successfully: {json.dumps(label_mapping, indent=2)}")
except Exception as e:
    print(f"Error loading label mapping: {str(e)}")
    exit(1)

BASE_URL = "http://127.0.0.1:5000"

def test_api_health():
    """Test the API health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        return True
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def test_predict():
    """Test the predict route with test data"""
    # Use a realistic example for better testing
    test_description = {
        "description": "This clinical trial investigates a new treatment for patients with mild to moderate Alzheimer's disease dementia. The trial evaluates whether the drug can slow cognitive decline and improve daily functioning."
    }
    headers = {'Content-Type': 'application/json'}
    
    print("\nCalling prediction API with test description:")
    print(f"Description: {test_description['description'][:100]}...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict", 
            json=test_description,
            headers=headers
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result.get('prediction')}")
            print(f"Confidence: {result.get('confidence', 0):.2%}")
            
            if 'all_scores' in result:
                print("\nConfidence scores for all classes:")
                for condition, score in sorted(result['all_scores'].items(), 
                                             key=lambda x: x[1], reverse=True):
                    print(f"  {condition}: {score:.2%}")
                    
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"Connection error: Make sure the Flask server is running at {BASE_URL}")
        return False
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Medical Trial Classifier API...")
    
    # Test health endpoint
    if not test_api_health():
        print("Health check failed, stopping tests.")
        exit(1)
        
    # Test prediction endpoint
    if not test_predict():
        print("Prediction test failed.")
        exit(1)
        
    print("\nAll tests passed successfully!")
