from flask import Flask, request, redirect, send_from_directory, jsonify, Response
import os
import json
import asyncio
from pathlib import Path
from functools import wraps

# Create Flask app for serving static files
app = Flask(__name__)

# Path to static directory
static_path = Path(__file__).parent / "static"

# Import orchestrator functions for direct API processing
from app.orchestrator import run_chain
from app.schemas import ValuationRequest, ValuationResponse

# Helper to run async functions in Flask
def async_handler(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(f(*args, **kwargs))
            return result
        finally:
            loop.close()
    return wrapper

# Serve the root page
@app.route('/')
def index():
    return send_from_directory(static_path, 'index.html')

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(static_path, path)

# API status endpoint
@app.route('/api/status')
def api_status():
    return jsonify({
        "status": "running",
        "message": "Farm Equipment Valuation API is available at /v2/value endpoint"
    })

# Valuation API endpoint
@app.route('/v2/value', methods=['POST'])
@async_handler
async def proxy_valuation():
    try:
        # Parse the request data
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        # Create a ValuationRequest model (with validation)
        try:
            val_request = ValuationRequest(
                make=payload.get('make', ''),
                model=payload.get('model', ''),
                year=int(payload.get('year', 0)),
                condition=payload.get('condition', ''),
                description=payload.get('description', '')
            )
        except Exception as e:
            return jsonify({"error": f"Invalid request data: {str(e)}"}), 400
        
        # Process the request using our orchestrator
        result_json = await run_chain(val_request.model_dump())
        
        # Parse and validate the response
        response_model = ValuationResponse.model_validate_json(result_json)
        
        # Return the response as JSON
        return jsonify(response_model.model_dump())
    except Exception as e:
        print(f"Valuation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": "Failed to calculate valuation"
        }), 500

# Catch-all route for other requests
@app.route('/<path:path>')
def catch_all(path):
    return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
