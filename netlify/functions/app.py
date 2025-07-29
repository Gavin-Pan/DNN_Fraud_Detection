import json
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

def handler(event, context):
    """Simple Netlify handler that returns JSON directly"""
    
    try:
        # Get the path and method
        path = event.get('path', '/')
        method = event.get('httpMethod', 'GET')
        
        # Remove Netlify function prefix
        if path.startswith('/.netlify/functions/app'):
            path = path.replace('/.netlify/functions/app', '') or '/'
        
        print(f"Processing: {method} {path}")
        
        # Handle CORS preflight
        if method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                },
                'body': ''
            }
        
        # Route handling
        if path == '/api/health' or path == '/health':
            return json_response({
                'status': 'healthy',
                'message': 'API is working!',
                'path': path,
                'method': method
            })
            
        elif path == '/api/test-transaction' or path == '/test-transaction':
            return json_response({
                'success': True,
                'sample_transactions': [
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
            })
            
        elif path == '/api/predict' or path == '/predict':
            if method != 'POST':
                return json_response({'error': 'Method not allowed'}, 405)
                
            try:
                # Get request body
                body = event.get('body', '{}')
                if isinstance(body, str):
                    data = json.loads(body)
                else:
                    data = body
                
                # Simple prediction logic (replace with your model later)
                prediction_result = {
                    'probability': 0.75,
                    'classification': 'SUSPICIOUS',
                    'risk_score': 7.5,
                    'explanation': 'High amount transfer detected'
                }
                
                return json_response({
                    'success': True,
                    'transaction_id': 'TXN_000001',
                    'prediction': prediction_result,
                    'input_data': data
                })
                
            except json.JSONDecodeError as e:
                return json_response({'error': f'Invalid JSON: {str(e)}'}, 400)
            except Exception as e:
                return json_response({'error': f'Prediction error: {str(e)}'}, 500)
                
        elif path == '/api/stats' or path == '/stats':
            return json_response({
                'success': True,
                'statistics': {
                    'total_predictions': 0,
                    'fraud_detected': 0,
                    'high_risk_detected': 0,
                    'fraud_rate_percent': 0.0,
                    'predictions_per_hour': 0.0,
                    'uptime_hours': 0.0,
                    'model_loaded': False
                }
            })
            
        elif path == '/api/model-info' or path == '/model-info':
            return json_response({
                'success': True,
                'model_info': {
                    'model_type': 'Deep Neural Network',
                    'version': '1.0.0',
                    'accuracy': 0.987,
                    'loaded': False
                }
            })
            
        else:
            return json_response({
                'error': 'Endpoint not found',
                'available_endpoints': [
                    '/api/health',
                    '/api/test-transaction', 
                    '/api/predict',
                    '/api/stats',
                    '/api/model-info'
                ]
            }, 404)
            
    except Exception as e:
        print(f"Handler error: {str(e)}")
        return json_response({
            'error': f'Server error: {str(e)}',
            'type': str(type(e).__name__)
        }, 500)

def json_response(data, status_code=200):
    """Helper function to create JSON responses with CORS headers"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        },
        'body': json.dumps(data)
    }

# Export for Netlify
def lambda_handler(event, context):
    return handler(event, context)