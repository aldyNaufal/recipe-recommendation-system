import logging
import os


class AppLogger:
    """Centralized logging configuration"""
    
    _logger = None
    
    @classmethod
    def get_logger(cls, name=__name__):
        """Get configured logger instance"""
        if cls._logger is None:
            cls.setup_logging()
        return logging.getLogger(name)
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('app.log')
            ]
        )
        cls._logger = logging.getLogger(__name__)
        return cls._logger