from flask import Flask, request, jsonify
import sys
import os

# Add the parent directory to Python path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing app
try:
    from app import app
except ImportError:
    # Fallback: create a simple Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return jsonify({
            "message": "FraudGuard API is running on Vercel!",
            "status": "healthy",
            "endpoints": {
                "/api/health": "Health check",
                "/api/predict": "Fraud prediction (POST)"
            }
        })
    
    @app.route('/api/health')
    def health():
        return jsonify({
            "status": "healthy",
            "platform": "vercel",
            "message": "Fraud detection API is running"
        })

# This is required for Vercel
handler = app