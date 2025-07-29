import sys
import os
import json

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# Import your existing Flask app
from app import app

def handler(event, context):
    """Netlify serverless function handler"""
    
    try:
        # Handle CORS preflight requests
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                },
                'body': ''
            }
        
        # Import serverless WSGI adapter
        try:
            import serverless_wsgi
            response = serverless_wsgi.handle_request(app, event, context)
        except ImportError:
            # Fallback if serverless-wsgi is not available
            response = fallback_handler(event, context)
            
        # Ensure CORS headers
        if 'headers' not in response:
            response['headers'] = {}
            
        response['headers']['Access-Control-Allow-Origin'] = '*'
        
        return response
        
    except Exception as e:
        # Error response
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'success': False,
                'error': f'Server error: {str(e)}',
                'debug': str(type(e).__name__)
            })
        }

def fallback_handler(event, context):
    """Fallback handler without serverless-wsgi"""
    try:
        # Extract request information from Netlify event
        method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        
        # Remove the Netlify functions prefix to get the actual path
        if path.startswith('/.netlify/functions/app'):
            path = path.replace('/.netlify/functions/app', '') or '/'
        
        headers = event.get('headers', {})
        body = event.get('body', '')
        query_string = event.get('queryStringParameters', {})
        
        # Create a test client
        with app.test_client() as client:
            # Prepare headers for the request
            request_headers = {}
            if 'content-type' in headers:
                request_headers['Content-Type'] = headers['content-type']
            
            # Make the request
            if method == 'GET':
                response = client.get(path, query_string=query_string, headers=request_headers)
            elif method == 'POST':
                response = client.post(path, data=body, headers=request_headers)
            elif method == 'PUT':
                response = client.put(path, data=body, headers=request_headers)
            elif method == 'DELETE':
                response = client.delete(path, headers=request_headers)
            else:
                response = client.open(path, method=method, data=body, headers=request_headers)
        
        # Convert Flask response to Netlify format
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With'
        }
        
        # Add Flask response headers
        for key, value in response.headers:
            response_headers[key] = value
        
        return {
            'statusCode': response.status_code,
            'headers': response_headers,
            'body': response.get_data(as_text=True)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'success': False,
                'error': f'Fallback handler error: {str(e)}'
            })
        }

# Export the handler - This is crucial!
def lambda_handler(event, context):
    return handler(event, context)