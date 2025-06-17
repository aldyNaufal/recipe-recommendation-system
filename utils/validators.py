from flask import current_app
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message, missing_fields=None, empty_fields=None, invalid_fields=None):
        self.message = message
        self.missing_fields = missing_fields or []
        self.empty_fields = empty_fields or []
        self.invalid_fields = invalid_fields or []
        super().__init__(self.message)

class RequestValidator:
    """Validator untuk request data"""
    
    @staticmethod
    def validate_request_data(data, required_fields=None, optional_fields=None):
        """Validate request data dengan field validation"""
        if not data:
            raise ValidationError("Request body is empty or invalid JSON")
        
        missing_fields = []
        empty_fields = []
        
        if required_fields:
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
                elif data[field] is None or str(data[field]).strip() == '':
                    empty_fields.append(field)
        
        if missing_fields or empty_fields:
            raise ValidationError(
                "Validation failed",
                missing_fields=missing_fields,
                empty_fields=empty_fields
            )
        
        return True
    
    @staticmethod
    def validate_user_id(user_id):
        """Validate user ID"""
        try:
            user_id = int(user_id)
            if user_id <= 0:
                raise ValidationError("User ID must be a positive integer")
            return user_id
        except (ValueError, TypeError):
            raise ValidationError("User ID must be a valid integer")
    
    @staticmethod
    def validate_top_k(top_k):
        """Validate top_k parameter"""
        try:
            top_k = int(top_k)
            max_recommendations = current_app.config.get('MAX_RECOMMENDATIONS', 50)
            default_recommendations = current_app.config.get('DEFAULT_RECOMMENDATIONS', 10)
            
            if top_k <= 0:
                return default_recommendations
            elif top_k > max_recommendations:
                return max_recommendations
            
            return top_k
        except (ValueError, TypeError):
            return current_app.config.get('DEFAULT_RECOMMENDATIONS', 10)
    
    @staticmethod
    def validate_categories(categories, recommender=None):
        """Validate categories list"""
        if not categories or not isinstance(categories, list):
            raise ValidationError("Categories must be a non-empty list")
        
        # Remove empty strings and duplicates
        valid_categories = list(set([cat.strip() for cat in categories if cat and cat.strip()]))
        
        if not valid_categories:
            raise ValidationError("At least one valid category must be provided")
        
        # Check against recommender's valid categories if available
        if recommender and hasattr(recommender, 'valid_categories'):
            available_categories = list(recommender.valid_categories)
            filtered_categories = [cat for cat in valid_categories if cat in available_categories]
            
            if not filtered_categories:
                raise ValidationError(
                    "No valid categories provided",
                    invalid_fields={
                        'provided_categories': valid_categories,
                        'available_categories': available_categories
                    }
                )
            
            return filtered_categories
        
        return valid_categories
    
    @staticmethod
    def validate_difficulty(difficulty, recommender=None):
        """Validate difficulty level"""
        if not difficulty:
            return None
        
        if recommender and hasattr(recommender, 'difficulty_mapping'):
            available_difficulties = list(recommender.difficulty_mapping.values())
            if difficulty not in available_difficulties:
                raise ValidationError(
                    "Invalid difficulty level",
                    invalid_fields={
                        'provided_difficulty': difficulty,
                        'available_difficulties': available_difficulties
                    }
                )
        
        return difficulty

class RecommendationValidator:
    """Validator khusus untuk recommendation requests"""
    
    @staticmethod
    def validate_existing_user_request(data):
        """Validate existing user recommendation request"""
        RequestValidator.validate_request_data(data, required_fields=['user_id'])
        
        user_id = RequestValidator.validate_user_id(data['user_id'])
        top_k = RequestValidator.validate_top_k(data.get('top_k', 10))
        
        return {
            'user_id': user_id,
            'top_k': top_k
        }
    
    @staticmethod
    def validate_new_user_request(data, recommender=None):
        """Validate new user recommendation request"""
        RequestValidator.validate_request_data(data, required_fields=['preferred_categories'])
        
        categories = RequestValidator.validate_categories(data['preferred_categories'], recommender)
        difficulty = RequestValidator.validate_difficulty(data.get('preferred_difficulty'), recommender)
        top_k = RequestValidator.validate_top_k(data.get('top_k', 10))
        
        return {
            'preferred_categories': categories,
            'preferred_difficulty': difficulty,
            'top_k': top_k
        }
    
    @staticmethod
    def validate_content_based_request(data, recommender=None):
        """Validate content-based recommendation request"""
        RequestValidator.validate_request_data(data)
        
        categories = None
        if 'preferred_categories' in data:
            categories = RequestValidator.validate_categories(data['preferred_categories'], recommender)
        
        top_k = RequestValidator.validate_top_k(data.get('top_k', 10))
        
        return {
            'preferred_categories': categories,
            'top_k': top_k
        }