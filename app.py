#!/usr/bin/env python3
"""
Fraud Detection Flask Application
Real-time fraud detection API using trained Deep Neural Network
"""

import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from datetime import datetime
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from models.fraud_model import FraudDetectionModel
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS for frontend-backend communication
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the fraud detection model
fraud_model = FraudDetectionModel()

# Statistics tracking
app_stats = {
    'total_predictions': 0,
    'fraud_detected': 0,
    'high_risk_detected': 0,
    'start_time': datetime.now(),
    'model_loaded': False
}


def initialize_model():
    """Load the fraud detection model on app startup"""
    try:
        logger.info("🚀 Initializing Fraud Detection System...")
        fraud_model.load_model()
        app_stats['model_loaded'] = True
        logger.info("✅ Fraud detection model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        app_stats['model_loaded'] = False

# ================================
# API ROUTES
# ================================

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    uptime = (datetime.now() - app_stats['start_time']).total_seconds()
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': app_stats['model_loaded'],
        'uptime_seconds': uptime,
        'total_predictions': app_stats['total_predictions'],
        'fraud_detected': app_stats['fraud_detected'],
        'version': '1.0.0'
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        model_info = fraud_model.get_model_info()
        return jsonify({
            'success': True,
            'model_info': model_info,
            'stats': app_stats
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    """
    Main fraud prediction endpoint
    
    Expected JSON payload:
    {
        "type": "TRANSFER",
        "amount": 250000,
        "oldbalanceOrg": 500000,
        "newbalanceOrig": 250000,
        "oldbalanceDest": 100000,
        "newbalanceDest": 350000,
        "isFlaggedFraud": 0,
        "step": 150
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        transaction_data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 'step'
        ]
        
        missing_fields = [field for field in required_fields if field not in transaction_data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Validate data types and ranges
        validation_result = validate_transaction_data(transaction_data)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': validation_result['error']
            }), 400
        
        # Make prediction
        logger.info(f"Processing fraud prediction for transaction: {transaction_data['type']} ${transaction_data['amount']}")
        
        prediction_result = fraud_model.predict_fraud_probability(transaction_data)
        
        # Update statistics
        app_stats['total_predictions'] += 1
        if prediction_result['classification'] == 'FRAUD':
            app_stats['fraud_detected'] += 1
        elif prediction_result['classification'] == 'SUSPICIOUS':
            app_stats['high_risk_detected'] += 1
        
        # Add transaction metadata
        response = {
            'success': True,
            'transaction_id': f"TXN_{app_stats['total_predictions']:06d}",
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_result,
            'model_info': {
                'version': '1.0.0',
                'model_type': fraud_model.get_model_info().get('model_type', 'Unknown')
            }
        }
        
        # Log the prediction
        log_prediction(transaction_data, prediction_result, response['transaction_id'])
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in fraud prediction: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple transactions"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({
                'success': False,
                'error': 'No transactions provided'
            }), 400
        
        if len(transactions) > 100:  # Limit batch size
            return jsonify({
                'success': False,
                'error': 'Batch size limited to 100 transactions'
            }), 400
        
        results = []
        
        for i, transaction in enumerate(transactions):
            try:
                # Validate transaction
                validation_result = validate_transaction_data(transaction)
                if not validation_result['valid']:
                    results.append({
                        'transaction_index': i,
                        'success': False,
                        'error': validation_result['error']
                    })
                    continue
                
                # Make prediction
                prediction_result = fraud_model.predict_fraud_probability(transaction)
                
                # Update stats
                app_stats['total_predictions'] += 1
                if prediction_result['classification'] == 'FRAUD':
                    app_stats['fraud_detected'] += 1
                
                results.append({
                    'transaction_index': i,
                    'success': True,
                    'transaction_id': f"TXN_{app_stats['total_predictions']:06d}",
                    'prediction': prediction_result
                })
                
            except Exception as e:
                results.append({
                    'transaction_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'batch_size': len(transactions),
            'processed': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get application statistics"""
    uptime = (datetime.now() - app_stats['start_time']).total_seconds()
    
    # Calculate rates
    fraud_rate = (app_stats['fraud_detected'] / max(app_stats['total_predictions'], 1)) * 100
    predictions_per_hour = (app_stats['total_predictions'] / max(uptime / 3600, 1))
    
    return jsonify({
        'success': True,
        'statistics': {
            'total_predictions': app_stats['total_predictions'],
            'fraud_detected': app_stats['fraud_detected'],
            'high_risk_detected': app_stats['high_risk_detected'],
            'fraud_rate_percent': round(fraud_rate, 2),
            'predictions_per_hour': round(predictions_per_hour, 2),
            'uptime_hours': round(uptime / 3600, 2),
            'model_loaded': app_stats['model_loaded'],
            'start_time': app_stats['start_time'].isoformat()
        }
    })

# ================================
# UTILITY FUNCTIONS
# ================================

def validate_transaction_data(data):
    """Validate transaction data"""
    try:
        # Check transaction type
        valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        if data.get('type') not in valid_types:
            return {
                'valid': False,
                'error': f'Invalid transaction type. Must be one of: {valid_types}'
            }
        
        # Check numeric fields
        numeric_fields = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
        for field in numeric_fields:
            try:
                value = float(data.get(field, 0))
                if value < 0:
                    return {
                        'valid': False,
                        'error': f'{field} cannot be negative'
                    }
                # Update the data with validated float value
                data[field] = value
            except (ValueError, TypeError):
                return {
                    'valid': False,
                    'error': f'{field} must be a valid number'
                }
        
        # Check boolean field
        is_flagged = data.get('isFlaggedFraud')
        if is_flagged not in [0, 1, True, False]:
            return {
                'valid': False,
                'error': 'isFlaggedFraud must be 0, 1, true, or false'
            }
        
        # Convert to integer
        data['isFlaggedFraud'] = int(is_flagged)
        
        # Business logic validation
        if data['amount'] <= 0:
            return {
                'valid': False,
                'error': 'Transaction amount must be greater than 0'
            }
        
        if data['amount'] > 10000000:  # 10M limit
            return {
                'valid': False,
                'error': 'Transaction amount exceeds maximum limit'
            }
        
        return {'valid': True}
        
    except Exception as e:
        return {
            'valid': False,
            'error': f'Validation error: {str(e)}'
        }

def log_prediction(transaction_data, prediction_result, transaction_id):
    """Log prediction for audit purposes"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': transaction_id,
            'transaction': transaction_data,
            'prediction': prediction_result
        }
        
        # In production, this would go to a proper logging system
        logger.info(f"PREDICTION: {transaction_id} - {prediction_result['classification']} "
                   f"(probability: {prediction_result['probability']:.3f})")
        
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

# ================================
# ERROR HANDLERS
# ================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

# ================================
# DEVELOPMENT HELPERS
# ================================

@app.route('/api/test-transaction', methods=['GET'])
def get_test_transaction():
    """Get sample transaction data for testing"""
    sample_transactions = [
        {
            'name': 'High Risk Transfer',
            'data': {
                'type': 'TRANSFER',
                'amount': 250000,
                'oldbalanceOrg': 500000,
                'newbalanceOrig': 250000,
                'oldbalanceDest': 100000,
                'newbalanceDest': 350000,
                'isFlaggedFraud': 0,
                'step': 150
            }
        },
        {
            'name': 'Low Risk Payment',
            'data': {
                'type': 'PAYMENT',
                'amount': 1000,
                'oldbalanceOrg': 50000,
                'newbalanceOrig': 49000,
                'oldbalanceDest': 0,
                'newbalanceDest': 1000,
                'isFlaggedFraud': 0,
                'step': 100
            }
        },
        {
            'name': 'Suspicious Cash Out',
            'data': {
                'type': 'CASH_OUT',
                'amount': 300000,
                'oldbalanceOrg': 300000,
                'newbalanceOrig': 0,
                'oldbalanceDest': 0,
                'newbalanceDest': 300000,
                'isFlaggedFraud': 0,
                'step': 200
            }
        }
    ]
    
    return jsonify({
        'success': True,
        'sample_transactions': sample_transactions
    })

# ================================
# MAIN APPLICATION #
# ================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    
    # Initialize model
    initialize_model()
    
    # Run the application
    print("🚀 Starting Fraud Detection API Server...")
    print(f"🌐 Access the application at: http://localhost:5000")
    print(f"📊 API Health Check: http://localhost:5000/api/health")
    print(f"🧪 Test Endpoint: http://localhost:5000/api/test-transaction")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.DEBUG
    )