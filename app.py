import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from config import config
from routes.recommendation_routes import recommendation_bp, init_routes
import traceback
from scripts.recipe_recommender import EnhancedIndonesianRecipeRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def create_app(config_name=None):
    """Application factory pattern"""
    
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app.config.from_object(config[config_name])
    
    # Configure CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize recommender - FIXED: Create instance, not just reference class
    recommender = None
    try:
        # Create an instance of the recommender
        recommender = EnhancedIndonesianRecipeRecommender()
        logger.info("Recommender instance created successfully")
        
        # Load the trained model and data
        # Use proper path joining and check current working directory
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # Try different possible paths
        possible_paths = [
            "models/recipe_recommender",  # Base name without extension
            "models/recipe_recommender_model.h5",  # Full path
            os.path.join(current_dir, "models", "recipe_recommender"),
            os.path.join(current_dir, "models", "recipe_recommender_model.h5")
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            # Check for the three required files
            base_path = model_path.replace("_model.h5", "").replace(".h5", "")
            h5_path = f"{base_path}_model.h5" if not base_path.endswith("_model.h5") else base_path
            components_path = f"{base_path}_components.joblib"
            metadata_path = f"{base_path}_metadata.json"
            
            logger.info(f"Checking paths:")
            logger.info(f"  H5 model: {h5_path} - exists: {os.path.exists(h5_path)}")
            logger.info(f"  Components: {components_path} - exists: {os.path.exists(components_path)}")
            logger.info(f"  Metadata: {metadata_path} - exists: {os.path.exists(metadata_path)}")
            
            if os.path.exists(h5_path) and os.path.exists(components_path):
                logger.info(f"Found model files, attempting to load from: {base_path}")
                
                # Use the load_model method that expects just the base path
                success = recommender.load_model(base_path)
                if success:
                    logger.info("Model loaded successfully!")
                    model_loaded = True
                    break
                else:
                    logger.warning(f"Failed to load model from {base_path}")
            
        if not model_loaded:
            # List all files in models directory for debugging
            models_dir = os.path.join(current_dir, "models")
            if os.path.exists(models_dir):
                files = os.listdir(models_dir)
                logger.warning(f"Files in models directory: {files}")
            else:
                logger.warning(f"Models directory does not exist: {models_dir}")
            
            logger.warning("Model files not found or failed to load. Recommender initialized but model not loaded.")
            
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {str(e)}")
        logger.error(traceback.format_exc())
        recommender = None
    
    # Initialize routes with recommender instance
    init_routes(recommender)
    
    # Register blueprints
    app.register_blueprint(recommendation_bp)
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found',
            'code': 'ENDPOINT_NOT_FOUND'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Global internal server error: {str(error)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        """Handle uncaught exceptions"""
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred',
            'code': 'UNEXPECTED_ERROR',
            'details': str(e) if app.debug else None
        }), 500
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Application health check"""
        try:
            status = {
                'success': True,
                'status': 'healthy',
                'service': 'recommendation_api',
                'version': '1.0.0',
                'environment': config_name,
                'recommender_loaded': recommender is not None
            }
            
            if recommender is not None:
                # Get detailed model status
                model_status = recommender.get_model_status()
                status.update({
                    'model_loaded': model_status['model_loaded'],
                    'components_status': model_status['components'],
                    'is_ready': recommender.is_model_loaded()
                })
            else:
                status.update({
                    'model_loaded': False,
                    'components_status': {},
                    'is_ready': False
                })
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return jsonify({
                'success': False,
                'status': 'error',
                'service': 'recommendation_api',
                'message': str(e)
            }), 500
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint"""
        return jsonify({
            'success': True,
            'message': 'Recommendation API Service',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'recommendations': '/api/recommendations/',
                'auth_required': True
            }
        })
    
    # Debug endpoint (only in development)
    @app.route('/debug/model', methods=['GET'])
    def debug_model():
        """Debug model loading status"""
        if config_name != 'development':
            return jsonify({'error': 'Debug endpoint only available in development'}), 403
            
        try:
            debug_info = {
                'current_directory': os.getcwd(),
                'recommender_exists': recommender is not None,
                'model_files_check': []
            }
            
            # Check model files
            models_dir = os.path.join(os.getcwd(), 'models')
            if os.path.exists(models_dir):
                files = os.listdir(models_dir)
                debug_info['models_directory_files'] = files
                
                # Check for specific model files
                expected_base_names = ['recipe_recommender', 'recipe_recommender_model']
                for base_name in expected_base_names:
                    h5_file = f"{base_name}_model.h5" if not base_name.endswith('_model') else f"{base_name}.h5"
                    components_file = f"{base_name}_components.joblib"
                    metadata_file = f"{base_name}_metadata.json"
                    
                    debug_info['model_files_check'].append({
                        'base_name': base_name,
                        'h5_exists': os.path.exists(os.path.join(models_dir, h5_file)),
                        'components_exists': os.path.exists(os.path.join(models_dir, components_file)),
                        'metadata_exists': os.path.exists(os.path.join(models_dir, metadata_file))
                    })
            else:
                debug_info['models_directory_files'] = 'Directory does not exist'
            
            if recommender is not None:
                debug_info['model_status'] = recommender.get_model_status()
            
            return jsonify(debug_info)
            
        except Exception as e:
            logger.error(f"Debug endpoint error: {str(e)}")
    # API info endpoint
    @app.route('/api', methods=['GET'])
    def api_info():
        """API information endpoint"""
        return jsonify({
            'success': True,
            'message': 'Recommendation API',
            'version': '1.0.0',
            'authentication': 'JWT Bearer Token required',
            'endpoints': {
                'existing_user_post': 'POST /api/recommendations/existing-user',
                'existing_user_get': 'GET /api/recommendations/existing-user/<user_id>',
                'new_user_post': 'POST /api/recommendations/new-user',
                'new_user_get': 'GET /api/recommendations/new-user-get',
                'content_based': 'POST /api/recommendations/content-based',
                'options': 'GET /api/recommendations/options',
                'health': 'GET /api/recommendations/health'
            }
        })
    
    # Request logging middleware
    @app.before_request
    def log_request_info():
        """Log request information"""
        if not request.path.startswith('/health'):  # Don't log health checks
            logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
            
            # Log auth header (without exposing token)
            auth_header = request.headers.get('Authorization')
            if auth_header:
                logger.info(f"Auth header present: Bearer ***")
            else:
                logger.info("No auth header")
    
    @app.after_request
    def log_response_info(response):
        """Log response information"""
        if not request.path.startswith('/health'):  # Don't log health checks
            logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
        return response
    
    logger.info(f"Flask app created with config: {config_name}")
    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    # Get configuration
    env = os.environ.get('FLASK_ENV', 'development')
    debug = env == 'development'
    
    logger.info(f"Starting Flask app in {env} mode")
    logger.info(f"Debug mode: {debug}")
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=debug
    )