import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    try:
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    except Exception as e:
        logger.warning(f"Could not serialize object {type(obj)}: {str(e)}")
        return str(obj)

class RecommendationSerializer:
    """Serializer untuk recommendation responses"""
    
    @staticmethod
    def serialize_recommendations(recommendations, user_id=None, user_type='existing', 
                                additional_data=None):
        """Serialize recommendations dengan format standar"""
        clean_recommendations = convert_to_serializable(recommendations)
        
        response_data = {
            'success': True,
            'total_recommendations': len(clean_recommendations),
            'recommendations': clean_recommendations,
            'user_type': user_type,
            'timestamp': datetime.now().isoformat()
        }
        
        if user_id:
            response_data['user_id'] = user_id
        
        if additional_data:
            response_data.update(additional_data)
        
        return response_data
    
    @staticmethod
    def serialize_error(error_message, error_code=None, details=None, status_code=500):
        """Serialize error response"""
        error_data = {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        if error_code:
            error_data['code'] = error_code
        
        if details:
            error_data['details'] = details
        
        return error_data, status_code
    
    @staticmethod
    def serialize_validation_error(missing_fields=None, empty_fields=None, invalid_fields=None):
        """Serialize validation error"""
        error_parts = []
        
        if missing_fields:
            error_parts.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        if empty_fields:
            error_parts.append(f"Empty fields not allowed: {', '.join(empty_fields)}")
        
        if invalid_fields:
            error_parts.append(f"Invalid fields: {', '.join(invalid_fields)}")
        
        error_message = '; '.join(error_parts)
        
        return RecommendationSerializer.serialize_error(
            error_message, 
            'VALIDATION_ERROR', 
            status_code=400
        )