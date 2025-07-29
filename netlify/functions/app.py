import json
import sys
import os
import traceback
from datetime import datetime

def handler(event, context):
    """Debug Netlify function to identify issues"""
    
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'event_path': event.get('path', 'NO_PATH'),
        'event_method': event.get('httpMethod', 'NO_METHOD'),
        'python_version': sys.version,
        'current_directory': os.getcwd(),
        'environment_vars': dict(os.environ),
        'sys_path': sys.path,
        'available_files': []
    }
    
    try:
        # List files in current directory
        debug_info['available_files'] = os.listdir('.')
    except Exception as e:
        debug_info['file_list_error'] = str(e)
    
    try:
        # Get the actual path being requested
        path = event.get('path', '/')
        method = event.get('httpMethod', 'GET')
        
        # Remove Netlify function prefix
        api_path = path
        if path.startswith('/.netlify/functions/app'):
            api_path = path.replace('/.netlify/functions/app', '') or '/'
        
        debug_info['processed_path'] = api_path
        
        # Handle CORS preflight
        if method == 'OPTIONS':
            return cors_response({'message': 'CORS preflight OK'})
        
        # Simple route handling without complex imports
        if api_path in ['/api/health', '/health', '/']:
            return cors_response({
                'status': 'healthy',
                'message': 'Netlify function is working!',
                'debug': debug_info
            })
            
        elif api_path in ['/api/test-transaction', '/test-transaction']:
            return cors_response({
                'success': True,
                'message': 'Test endpoint working!',
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
                    }
                ]
            })
            
        elif api_path in ['/api/predict', '/predict']:
            if method != 'POST':
                return cors_response({'error': 'Method not allowed'}, 405)
                
            try:
                # Parse request body
                body = event.get('body', '{}')
                if isinstance(body, str):
                    data = json.loads(body) if body else {}
                else:
                    data = body or {}
                
                # Mock prediction (replace with real model later)
                amount = data.get('amount', 0)
                transaction_type = data.get('type', 'UNKNOWN')
                
                # Simple risk scoring
                if amount > 100000:
                    risk_score = 8.5
                    classification = 'HIGH_RISK'
                    probability = 0.85
                elif amount > 50000:
                    risk_score = 6.0
                    classification = 'SUSPICIOUS' 
                    probability = 0.60
                else:
                    risk_score = 2.0
                    classification = 'LOW_RISK'
                    probability = 0.20
                
                return cors_response({
                    'success': True,
                    'transaction_id': f'TXN_{datetime.now().strftime("%Y%m%d%H%M%S")}',
                    'prediction': {
                        'probability': probability,
                        'classification': classification,
                        'risk_score': risk_score,
                        'explanation': f'{transaction_type} transaction of ${amount:,}'
                    },
                    'input_data': data
                })
                
            except json.JSONDecodeError as e:
                return cors_response({'error': f'Invalid JSON: {str(e)}'}, 400)
            except Exception as e:
                return cors_response({'error': f'Prediction error: {str(e)}'}, 500)
                
        elif api_path in ['/api/stats', '/stats']:
            return cors_response({
                'success': True,
                'statistics': {
                    'total_predictions': 0,
                    'fraud_detected': 0,
                    'high_risk_detected': 0,
                    'fraud_rate_percent': 0.0,
                    'predictions_per_hour': 0.0,
                    'uptime_hours': 0.0,
                    'model_loaded': False,
                    'start_time': datetime.now().isoformat()
                }
            })
            
        elif api_path in ['/api/model-info', '/model-info']:
            return cors_response({
                'success': True,
                'model_info': {
                    'model_type': 'Deep Neural Network (Mock)',
                    'version': '1.0.0',
                    'accuracy': 0.987,
                    'loaded': False,
                    'note': 'This is a mock response - model not loaded yet'
                }
            })
            
        else:
            return cors_response({
                'error': 'Endpoint not found',
                'requested_path': api_path,
                'available_endpoints': [
                    '/api/health',
                    '/api/test-transaction', 
                    '/api/predict',
                    '/api/stats',
                    '/api/model-info'
                ],
                'debug': debug_info
            }, 404)
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        return cors_response({
            'error': f'Function error: {str(e)}',
            'traceback': error_traceback,
            'debug': debug_info
        }, 500)

def cors_response(data, status_code=200):
    """Create a JSON response with CORS headers"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
            'Cache-Control': 'no-cache'
        },
        'body': json.dumps(data, indent=2)
    }

# Export the handler for Netlify
def lambda_handler(event, context):
    return handler(event, context)