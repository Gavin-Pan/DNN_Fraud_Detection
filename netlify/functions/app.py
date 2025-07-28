import sys
import os
import json

# Add the project root to Python path so we can import everything
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# Import your existing Flask app
from app import app

def handler(event, context):
    """
    Netlify serverless function handler for your Flask app with TensorFlow model
    """
    
    try:
        # Import serverless WSGI adapter
        import serverless_wsgi
        
        # Handle CORS preflight requests
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Access-Control-Max-Age': '86400'
                },
                'body': ''
            }
        
        # Use serverless-wsgi to handle the Flask app
        response = serverless_wsgi.handle_request(app, event, context)
        
        # Ensure CORS headers are always present
        if 'headers' not in response:
            response['headers'] = {}
            
        response['headers'].update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With'
        })
        
        return response
        
    except ImportError:
        # Fallback if serverless-wsgi not available
        return fallback_handler(event, context)
    except Exception as e:
        # Error handling
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'success': False,
                'error': f'Server error: {str(e)}',
                'message': 'Flask app encountered an error'
            })
        }

def fallback_handler(event, context):
    """Fallback handler without serverless-wsgi"""
    try:
        from werkzeug.test import Client
        from werkzeug.wrappers import Response
        
        # Create a test client for your Flask app
        client = Client(app, Response)
        
        # Extract request information from Netlify event
        method = event.get('httpMethod', 'GET')
        path = event.get('path', '/').replace('/.netlify/functions/app', '')
        if not path or path == '/':
            path = '/'
        
        headers = event.get('headers', {})
        body = event.get('body', '')
        query_string = event.get('rawQuery', '')
        
        # Prepare the path with query string
        if query_string:
            path = f"{path}?{query_string}"
        
        # Make request to Flask app
        if method in ['POST', 'PUT', 'PATCH']:
            response = client.open(
                path, 
                method=method, 
                data=body,
                headers=[(k, v) for k, v in headers.items()],
                content_type=headers.get('content-type', 'application/json')
            )
        else:
            response = client.open(
                path, 
                method=method,
                headers=[(k, v) for k, v in headers.items()]
            )
        
        # Convert Flask response to Netlify format
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With'
        }
        
        # Add Flask response headers
        for header, value in response.headers:
            response_headers[header] = value
        
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

# Export the handler
app_handler = handler