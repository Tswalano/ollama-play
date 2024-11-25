from functools import wraps
from flask import jsonify
from app.utils.logger import logger

def handle_errors(f):
    """Error handling decorator for API routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return decorated_function