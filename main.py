"""
Flask API for Medical Trial Classification with PubMedBERT
Author: Harry Patria
"""

import os
import json
import torch
import nltk
import logging
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from werkzeug.middleware.proxy_fix import ProxyFix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup NLTK
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
    
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
    nltk.data.path.append(nltk_data_path)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.warning(f"Warning when downloading NLTK data: {str(e)}")

app = Flask(__name__, static_folder='static', template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app)  # Fix for proxy headers
CORS(app)  # Enable CORS for all routes

# Path to the saved model
MODEL_DIR = os.environ.get("MODEL_DIR", "deployment/pubmedbert")
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
                logger.info(f"Label mapping loaded with {len(id2label)} classes")
        else:
            logger.error(f"Label mapping file not found at {LABEL_FILE}")
            return False
        
        # Load model and tokenizer
        try:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model.to(device)
            model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully on {device}")
            return True
        except Exception as e:
            logger.error(f"Error loading model/tokenizer: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
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
        logger.info("Model not loaded, attempting to load now")
        success = load_model()
        if not success:
            logger.error("Failed to load model")
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
        
        # Log prediction information
        logger.info(f"Prediction: {prediction}, Confidence: {float(probabilities[0][pred_idx]):.4f}")
        
        return {
            "prediction": prediction,
            "confidence": float(probabilities[0][pred_idx]),
            "all_scores": confidence_scores
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
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
    model_loaded = model is not None
    tokenizer_loaded = tokenizer is not None
    
    # If model isn't loaded yet, try loading it
    if not model_loaded:
        try:
            model_loaded = load_model()
        except Exception as e:
            logger.error(f"Error loading model during health check: {str(e)}")
    
    status = "healthy" if model_loaded and tokenizer_loaded else "degraded"
    
    return jsonify({
        "status": status, 
        "model_loaded": model_loaded,
        "tokenizer_loaded": tokenizer_loaded,
        "device": str(device),
        "api_version": "1.1.0"
    })

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
            logger.warning("Request was not JSON")
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        
        # Check if description is provided
        if "description" not in data:
            logger.warning("No description provided in request")
            return jsonify({"error": "No description provided in request"}), 400
        
        description = data["description"]
        
        # Check if description is empty or too short
        if not description or len(description.strip()) < 10:
            logger.warning("Description too short")
            return jsonify({"error": "Description is too short. Please provide at least 10 characters."}), 400
        
        # Get the prediction
        result = predict(description)
        
        # Return the prediction
        return jsonify(result)
    
    except ValueError as ve:
        # Handle specific validation errors
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
        
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error processing request: {str(e)}")
        logger.error(error_trace)
        
        # Return an error response
        return jsonify({"error": str(e)}), 500

@app.route("/api/model_info")
def model_info():
    """Endpoint to provide information about the model."""
    if not id2label:
        load_model()
    
    return jsonify({
        "model_type": "PubMedBERT",
        "model_size": "110M parameters",
        "labels": list(id2label.values()) if id2label else [],
        "label_count": len(id2label) if id2label else 0,
        "max_sequence_length": 512
    })

@app.errorhandler(404)
def page_not_found(e):
    """Handler for 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handler for 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == "__main__":
    # Load the model when the application starts
    logger.info("Starting application...")
    logger.info(f"Model directory: {MODEL_DIR}")
    
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Start the Flask server
    logger.info(f"Starting server on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
