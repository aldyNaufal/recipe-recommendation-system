import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'recipe-recommender-secret-key-2024'
    DEBUG = False
    TESTING = False
    
    # API Configuration
    API_VERSION = '1.0.0'
    API_TITLE = 'Indonesian Recipe Recommendation System'
    
    # Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH') or "models/fix_model.pkl"
    DATA_PATH = os.environ.get('DATA_PATH') or "data/data_recipes_cleaned.csv"
    
    # Recommendation Limits
    MAX_RECOMMENDATIONS = 50
    DEFAULT_RECOMMENDATIONS = 10
    MAX_SEARCH_RESULTS = 50
    DEFAULT_SEARCH_RESULTS = 20
    
    # Rating Configuration
    MIN_RATING_DEFAULT = 3.0
    MAX_DIFFICULTY_DEFAULT = 3
    
    # CORS Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.environ.get('LOG_FILE') or 'logs/app.log'
    
    # Server Configuration
    HOST = os.environ.get('HOST') or '0.0.0.0'
    PORT = int(os.environ.get('PORT') or 5000)
    
    # Cache Configuration (jika diperlukan di masa depan)
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Rate Limiting (jika diperlukan)
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'memory://'
    RATELIMIT_DEFAULT = "100 per hour"
    
    @staticmethod
    def init_app(app):
        """Initialize application configuration"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Development specific settings
    FLASK_ENV = 'development'
    
    # Relaxed limits for development
    MAX_RECOMMENDATIONS = 100
    MAX_SEARCH_RESULTS = 100
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        
        # Development specific initialization
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format=Config.LOG_FORMAT
        )

class ProductionConfig(Config):
    """Production configuration"""
    
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Stricter limits for production
    MAX_RECOMMENDATIONS = 50
    MAX_SEARCH_RESULTS = 50
    
    # Rate limiting for production
    RATELIMIT_DEFAULT = "1000 per hour"
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        
        # Production logging setup
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not os.path.exists('logs'):
            os.mkdir('logs')
            
        file_handler = RotatingFileHandler(
            Config.LOG_FILE, 
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        file_handler.setLevel(logging.WARNING)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.WARNING)

class TestingConfig(Config):
    """Testing configuration"""
    
    TESTING = True
    DEBUG = True
    
    # Test database/model paths
    MODEL_PATH = "tests/fixtures/test_model.pkl"
    DATA_PATH = "tests/fixtures/test_data.csv"
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # Fast cache for testing
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 1
    
    # Relaxed rate limits for testing
    RATELIMIT_DEFAULT = "1000 per minute"
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, DevelopmentConfig)