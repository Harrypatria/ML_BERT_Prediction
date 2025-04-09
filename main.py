"""
Flask API for Medical Trial Classification with PubMedBERT
Author: Harry Patria
"""

import os
import json
import torch
import nltk
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup NLTK
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
    
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
    nltk.data.path.append(nltk_data_path)
except Exception as e:
    print(f"Warning when downloading NLTK data: {str(e)}")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Path to the saved model
MODEL_DIR = "deployment/pubmedbert"
LABEL_FILE = os.path.join(MODEL_DIR, "label_mapping.json")

# Global variables to store model and preprocessing tools
model = None
tokenizer = None
id2label = None
label2id = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the transformer model and its associated files."""
    global model, tokenizer, id2label, label2id
    
    try:
        # Load label mapping
        if os.path.exists(LABEL_FILE):
            with open(LABEL_FILE, 'r') as f:
                label_mapping = json.load(f)
                id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
                label2id = label_mapping['label2id']
                print(f"Label mapping loaded: {id2label}")
        else:
            print(f"Label mapping file not found at {LABEL_FILE}")
            return False
        
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def predict(description: str) -> dict:
    """
    Function that takes in the description text and returns the prediction.
    
    Args:
        description: Text description of a medical trial
    
    Returns:
        Dictionary with prediction and confidence scores
    """
    global model, tokenizer, id2label, device
    
    if not model or not tokenizer:
        success = load_model()
        if not success:
            raise ValueError("Failed to load model")
    
    try:
        # Prepare input
        inputs = tokenizer(
            description,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get predictions with confidence scores
        pred_idx = torch.argmax(probabilities, dim=1).item()
        prediction = id2label[pred_idx]
        
        # Get confidence scores for all classes
        confidence_scores = {id2label[i]: float(prob) for i, prob in enumerate(probabilities[0])}
        
        return {
            "prediction": prediction,
            "confidence": float(probabilities[0][pred_idx]),
            "all_scores": confidence_scores
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

@app.route('/')
def index():
    """Render the main page with the interactive UI."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/api/predict", methods=["POST"])
def identify_condition():
    """
    Endpoint to predict the medical condition based on a trial description.
    
    Expected JSON input: {"description": "description text"}
    Returns: {"prediction": "predicted label", "confidence": 0.95, ...}
    """
    try:
        # Parse the request data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        
        # Check if description is provided
        if "description" not in data:
            return jsonify({"error": "No description provided in request"}), 400
        
        # Get the prediction
        result = predict(data["description"])
        
        # Return the prediction
        return jsonify(result)
    
    except Exception as e:
        # Log the error for debugging
        import traceback
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        
        # Return an error response
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Load the model when the application starts
    print("Loading model...")
    load_model()
    
    # Start the Flask server
    print("Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
