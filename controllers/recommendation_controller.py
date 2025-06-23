from flask import request, jsonify, current_app
from services.recommendation_service import RecommendationService
from utils.serializers import RecommendationSerializer
import logging

logger = logging.getLogger(__name__)

class RecommendationController:
    """Controller untuk handling recommendation requests"""
    
    def __init__(self, recommender=None):
        self.recommendation_service = RecommendationService(recommender)
    
    def set_recommender(self, recommender):
        """Set recommender model"""
        self.recommendation_service.set_recommender(recommender)

    def _check_model_loaded(self):
        """Helper method untuk check model"""
        if not self.recommendation_service.recommender:
            return False, "Model not loaded"
        
        recommender = self.recommendation_service.recommender
        
        # Gunakan is_model_loaded() method
        if hasattr(recommender, 'is_model_loaded') and callable(recommender.is_model_loaded):
            if recommender.is_model_loaded():
                return True, "Model loaded successfully"
            else:
                return False, "Model components not fully loaded"
        else:
            # Fallback manual check
            if (hasattr(recommender, 'model') and recommender.model is not None and
                hasattr(recommender, 'encoders') and recommender.encoders is not None and
                hasattr(recommender, 'processed_data') and recommender.processed_data is not None):
                return True, "Model loaded successfully"
            else:
                return False, "Model components missing"  
    
    def get_existing_user_recommendations_post(self):
        """Handle POST request for existing user recommendations"""
        try:
            # Check if model is loaded
            if not self.recommendation_service.recommender:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Model not loaded',
                    error_code='MODEL_NOT_LOADED'
                )), 500
            
            # Check Content-Type
            if not request.is_json:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Content-Type must be application/json',
                    error_code='INVALID_CONTENT_TYPE'
                )), 400
            
            # Get request data
            data = request.get_json()
            user_info = getattr(request, 'current_user', None)
            
            logger.info(f"POST existing user recommendations request: {data}")
            logger.info(f"Authenticated user: {user_info}")
            
            # Process request
            result = self.recommendation_service.get_existing_user_recommendations(data, user_info)
            
            # Check if result is error tuple
            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Controller error in existing user recommendations POST: {str(e)}")
            error_data = RecommendationSerializer.serialize_error(
                'Internal server error',
                error_code='INTERNAL_ERROR',
                details=str(e) if current_app.debug else None
            )
            return jsonify(error_data[0]), error_data[1]
    
    def get_existing_user_recommendations_get(self, user_id):
        """Handle GET request for existing user recommendations"""
        try:
            # Check if model is loaded
            if not self.recommendation_service.recommender:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Model not loaded',
                    error_code='MODEL_NOT_LOADED'
                )), 500
            
            # Get query parameters
            top_k = request.args.get('top_k', 10, type=int)
            user_info = getattr(request, 'current_user', None)
            
            # Create data dict
            data = {
                'user_id': user_id,
                'top_k': top_k
            }
            
            logger.info(f"GET existing user recommendations request: {data}")
            logger.info(f"Authenticated user: {user_info}")
            
            # Process request
            result = self.recommendation_service.get_existing_user_recommendations(data, user_info)
            
            # Check if result is error tuple
            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Controller error in existing user recommendations GET: {str(e)}")
            error_data = RecommendationSerializer.serialize_error(
                'Internal server error',
                error_code='INTERNAL_ERROR',
                details=str(e) if current_app.debug else None
            )
            return jsonify(error_data[0]), error_data[1]
    
    def get_new_user_recommendations_post(self):
        """Handle POST request for new user recommendations"""
        try:
            # Check if model is loaded
            if not self.recommendation_service.recommender:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Model not loaded',
                    error_code='MODEL_NOT_LOADED'
                )), 500
            
            # Check Content-Type
            if not request.is_json:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Content-Type must be application/json',
                    error_code='INVALID_CONTENT_TYPE'
                )), 400
            
            # Get request data
            data = request.get_json()
            user_info = getattr(request, 'current_user', None)
            
            logger.info(f"POST new user recommendations request: {data}")
            logger.info(f"Authenticated user: {user_info}")
            
            # Process request
            result = self.recommendation_service.get_new_user_recommendations(data, user_info)
            
            # Check if result is error tuple
            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Controller error in new user recommendations POST: {str(e)}")
            error_data = RecommendationSerializer.serialize_error(
                'Internal server error',
                error_code='INTERNAL_ERROR',
                details=str(e) if current_app.debug else None
            )
            return jsonify(error_data[0]), error_data[1]
    
    def get_new_user_recommendations_get(self):
        """Handle GET request for new user recommendations with query parameters"""
        try:
            # Check if model is loaded
            if not self.recommendation_service.recommender:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Model not loaded',
                    error_code='MODEL_NOT_LOADED'
                )), 500
            
            # Get categories from query params (comma-separated)
            categories_param = request.args.get('categories', '')
            if not categories_param:
                return jsonify(RecommendationSerializer.serialize_error(
                    'categories parameter is required (comma-separated list)',
                    error_code='MISSING_CATEGORIES'
                )), 400
            
            preferred_categories = [cat.strip() for cat in categories_param.split(',') if cat.strip()]
            preferred_difficulty = request.args.get('difficulty')
            top_k = request.args.get('top_k', 10, type=int)
            user_info = getattr(request, 'current_user', None)
            
            # Create data dict
            data = {
                'preferred_categories': preferred_categories,
                'preferred_difficulty': preferred_difficulty,
                'top_k': top_k
            }
            
            logger.info(f"GET new user recommendations request: {data}")
            logger.info(f"Authenticated user: {user_info}")
            
            # Process request
            result = self.recommendation_service.get_new_user_recommendations(data, user_info)
            
            # Check if result is error tuple
            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Controller error in new user recommendations GET: {str(e)}")
            error_data = RecommendationSerializer.serialize_error(
                'Internal server error',
                error_code='INTERNAL_ERROR',
                details=str(e) if current_app.debug else None
            )
            return jsonify(error_data[0]), error_data[1]
    
    def get_content_based_recommendations_post(self):
        """Handle POST request for content-based recommendations"""
        try:
            # Check if model is loaded
            if not self.recommendation_service.recommender:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Model not loaded',
                    error_code='MODEL_NOT_LOADED'
                )), 500
            
            # Check Content-Type
            if not request.is_json:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Content-Type must be application/json',
                    error_code='INVALID_CONTENT_TYPE'
                )), 400
            
            # Get request data
            data = request.get_json()
            user_info = getattr(request, 'current_user', None)
            
            logger.info(f"POST content-based recommendations request: {data}")
            logger.info(f"Authenticated user: {user_info}")
            
            # Process request
            result = self.recommendation_service.get_content_based_recommendations(data, user_info)
            
            # Check if result is error tuple
            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Controller error in content-based recommendations: {str(e)}")
            error_data = RecommendationSerializer.serialize_error(
                'Internal server error',
                error_code='INTERNAL_ERROR',
                details=str(e) if current_app.debug else None
            )
            return jsonify(error_data[0]), error_data[1]
    
    def get_recommendation_options(self):
        """Handle request for recommendation options"""
        try:
            user_info = getattr(request, 'current_user', None)
            logger.info(f"Get recommendation options request from user: {user_info}")
            
            result = self.recommendation_service.get_recommendation_options()
            
            # Check if result is error tuple
            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Controller error in get recommendation options: {str(e)}")
            error_data = RecommendationSerializer.serialize_error(
                'Internal server error',
                error_code='INTERNAL_ERROR',
                details=str(e) if current_app.debug else None
            )
            return jsonify(error_data[0]), error_data[1]
    
    def test_new_user_sample(self):
        """Test endpoint with sample data for new user"""
        try:
            sample_data = {
                'preferred_categories': ['Indonesian', 'Asian', 'Vegetarian'],
                'preferred_difficulty': 'Easy',
                'top_k': 5
            }
            
            user_info = getattr(request, 'current_user', None)
            logger.info(f"Test new user sample request from user: {user_info}")
            
            if not self.recommendation_service.recommender:
                return jsonify(RecommendationSerializer.serialize_error(
                    'Model not loaded',
                    error_code='MODEL_NOT_LOADED'
                )), 500
            
            result = self.recommendation_service.get_new_user_recommendations(sample_data, user_info)
            
            # Check if result is error tuple
            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]
            
            # Modify result for test endpoint
            if 'recommendations' in result:
                result['recommendations'] = result['recommendations'][:3]  # Show only first 3
            
            result['message'] = 'Sample new user recommendations'
            result['sample_input'] = sample_data
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Controller error in test new user sample: {str(e)}")
            error_data = RecommendationSerializer.serialize_error(
                'Internal server error',
                error_code='INTERNAL_ERROR',
                details=str(e) if current_app.debug else None
            )
            return jsonify(error_data[0]), error_data[1]