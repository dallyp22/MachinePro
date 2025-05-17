from dotenv import load_dotenv
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Check required API keys
required_vars = ['OPENAI_API_KEY', 'OPENAI_VECTOR_STORE_ID']
missing_vars = [var for var in required_vars if not os.environ.get(var)]

if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error("Please set these variables in your environment or .env file")
    sys.exit(1)

# Print environment variables for debugging (without showing full values)
api_key = os.environ.get('OPENAI_API_KEY')
vector_store_id = os.environ.get('OPENAI_VECTOR_STORE_ID')

logger.info(f"OPENAI_API_KEY: {'*' * 8}{api_key[-5:] if api_key else 'Not set'}")
logger.info(f"OPENAI_VECTOR_STORE_ID: {'*' * 8}{vector_store_id[-5:] if vector_store_id else 'Not set'}")

try:
    # Import the Flask app
    from wsgi import app
    
    # This is what gunicorn will use
    if __name__ == "__main__":
        logger.info("Starting AgIQ v2 Farm Equipment Valuation System")
        app.run(host="0.0.0.0", port=5000)
except Exception as e:
    logger.error(f"Error starting application: {str(e)}")
    sys.exit(1)
