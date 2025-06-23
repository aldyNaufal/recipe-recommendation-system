# Di core/app_factory.py
import os
from flask import Flask
from flask_cors import CORS
from config import config
from routes.recommendation_routes import recommendation_bp, init_routes
from core.logger import AppLogger
from core.recommender_init import RecommenderInitializer
from handlers.error_handlers import ErrorHandlers
from handlers.middleware import Middleware
from handlers.utility_routes import UtilityRoutes

class FlaskAppFactory:
    """Flask application factory"""
    
    def __init__(self):
        self.logger = AppLogger.get_logger(__name__)
    
    def create_app(self, config_name=None):
        """Create and configure Flask application"""
        # Create Flask app
        app = Flask(__name__)
        
        # Configure app
        config_name = config_name or os.environ.get('FLASK_ENV', 'development')
        app.config.from_object(config[config_name])
        
        # Setup CORS dengan konfigurasi yang lebih lengkap
        CORS(app, 
             origins=app.config['CORS_ORIGINS'],
             methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
             allow_headers=['Content-Type', 'Authorization'],
             supports_credentials=True)
        
        # Log CORS configuration
        self.logger.info(f"CORS Origins: {app.config['CORS_ORIGINS']}")
        
        # Initialize recommender
        recommender_init = RecommenderInitializer()
        recommender = recommender_init.initialize_recommender()
        
        # Initialize routes
        init_routes(recommender)
        app.register_blueprint(recommendation_bp)
        
        # Register handlers and middleware
        self._register_components(app, config_name, recommender)
        
        self.logger.info(f"Flask app created with config: {config_name}")
        return app
    
    def _register_components(self, app, config_name, recommender):
        """Register all app components"""
        # Error handlers
        error_handlers = ErrorHandlers()
        error_handlers.register_handlers(app)
        
        # Middleware
        middleware = Middleware()
        middleware.register_middleware(app)
        
        # Utility routes
        utility_routes = UtilityRoutes()
        utility_routes.register_routes(app, config_name, recommender)