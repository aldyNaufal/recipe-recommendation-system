import traceback
from flask import jsonify
from core.logger import AppLogger


class ErrorHandlers:
    """Global error handlers for Flask application"""
    
    def __init__(self):
        self.logger = AppLogger.get_logger(__name__)
    
    def register_handlers(self, app):
        """Register all error handlers with Flask app"""
        
        @app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'success': False,
                'error': 'Endpoint not found',
                'code': 'ENDPOINT_NOT_FOUND'
            }), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            self.logger.error(f"Global internal server error: {str(error)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'code': 'INTERNAL_ERROR'
            }), 500
        
        @app.errorhandler(Exception)
        def handle_exception(e):
            """Handle uncaught exceptions"""
            self.logger.error(f"Unhandled exception: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return jsonify({
                'success': False,
                'error': 'An unexpected error occurred',
                'code': 'UNEXPECTED_ERROR',
                'details': str(e) if app.debug else None
            }), 500