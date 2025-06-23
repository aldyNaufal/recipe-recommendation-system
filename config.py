import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or '5799f39d3045afe7aacdeac7e65888e53e1efb824b9c46a6c306dd5b71a89a488ada8049ec4c4736cb064877ea345c4c0b0449adad92e28124739351db7c2799'
    JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM') or 'HS256'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Recommendation settings
    MAX_RECOMMENDATIONS = int(os.environ.get('MAX_RECOMMENDATIONS', '50'))
    DEFAULT_RECOMMENDATIONS = int(os.environ.get('DEFAULT_RECOMMENDATIONS', '10'))
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', './models/')
    
    # CORS settings
    # CORS settings
    @staticmethod
    def get_cors_origins():
        cors_env = os.environ.get('CORS_ORIGINS')
        if cors_env:
            return [origin.strip() for origin in cors_env.split(',')]
        else:
            # Default origins dengan berbagai variasi
            return [
                'http://localhost:5173',
                'http://localhost:3000', 
                'http://127.0.0.1:5173',
                'http://127.0.0.1:3000'
            ]
    
    CORS_ORIGINS = get_cors_origins()
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


