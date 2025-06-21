from flask import request
from core.logger import AppLogger


class Middleware:
    """Request and response middleware"""
    
    def __init__(self):
        self.logger = AppLogger.get_logger(__name__)
    
    def register_middleware(self, app):
        """Register middleware with Flask app"""
        
        @app.before_request
        def log_request_info():
            """Log request information"""
            if not request.path.startswith('/health'):
                self.logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
                
                auth_header = request.headers.get('Authorization')
                if auth_header:
                    self.logger.info("Auth header present: Bearer ***")
                else:
                    self.logger.info("No auth header")
        
        @app.after_request
        def log_response_info(response):
            """Log response information"""
            if not request.path.startswith('/health'):
                self.logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
            return response