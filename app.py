import os
from core.app_factory import FlaskAppFactory
from core.logger import AppLogger


def create_app(config_name=None):
    """Main application factory function"""
    factory = FlaskAppFactory()
    return factory.create_app(config_name)


# Create app instance
app = create_app()

if __name__ == '__main__':
    # Get configuration
    env = os.environ.get('FLASK_ENV', 'development')
    debug = env == 'development'
    
    logger = AppLogger.get_logger(__name__)
    logger.info(f"Starting Flask app in {env} mode")
    logger.info(f"Debug mode: {debug}")
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=debug
    )