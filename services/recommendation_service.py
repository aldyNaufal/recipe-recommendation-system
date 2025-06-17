import logging
from utils.validators import RecommendationValidator, ValidationError
from utils.serializers import RecommendationSerializer
from datetime import datetime

logger = logging.getLogger(__name__)

class RecommendationService:
    """Service untuk business logic recommendations"""
    
    def __init__(self, recommender=None):
        self.recommender = recommender
    
    def set_recommender(self, recommender):
        """Set recommender model"""
        self.recommender = recommender
    
    def get_existing_user_recommendations(self, data, user_info=None):
        """Get recommendations for existing user"""
        try:
            # Validate request
            validated_data = RecommendationValidator.validate_existing_user_request(data)
            
            user_id = validated_data['user_id']
            top_k = validated_data['top_k']
            
            # Optional: Verify user_id matches authenticated user
            if user_info and user_info.get('user_id'):
                auth_user_id = user_info['user_id']
                if user_id != auth_user_id:
                    logger.warning(f"User {auth_user_id} requesting recommendations for user {user_id}")
                    # You can choose to allow or deny this based on your business logic
            
            logger.info(f"Getting recommendations for user: {user_id}, top_k: {top_k}")
            
            # Get recommendations from model
            recommendations = self.recommender.get_enhanced_recommendations(
                user_id=user_id,
                top_k=top_k,
                show_detailed=False
            )
            
            # Serialize response
            additional_data = {
                'recommendation_method': 'collaborative_filtering'
            }
            
            if user_info:
                additional_data['authenticated_user'] = user_info['user_id']
            
            return RecommendationSerializer.serialize_recommendations(
                recommendations=recommendations,
                user_id=user_id,
                user_type='existing',
                additional_data=additional_data
            )
            
        except ValidationError as e:
            return RecommendationSerializer.serialize_validation_error(
                missing_fields=e.missing_fields,
                empty_fields=e.empty_fields,
                invalid_fields=e.invalid_fields
            )
        except Exception as e:
            logger.error(f"Error in existing user recommendations: {str(e)}")
            return RecommendationSerializer.serialize_error(
                "Failed to get recommendations",
                error_code="RECOMMENDATION_ERROR",
                details=str(e)
            )
    
    def get_new_user_recommendations(self, data, user_info=None):
        """Get recommendations for new user based on preferences"""
        try:
            # Validate request
            validated_data = RecommendationValidator.validate_new_user_request(data, self.recommender)
            
            preferred_categories = validated_data['preferred_categories']
            preferred_difficulty = validated_data['preferred_difficulty']
            top_k = validated_data['top_k']
            
            logger.info(f"Getting new user recommendations with categories: {preferred_categories}, "
                       f"difficulty: {preferred_difficulty}, top_k: {top_k}")
            
            # Get recommendations from model
            recommendations = self.recommender.get_user_profile_based_recommendations(
                preferred_categories=preferred_categories,
                preferred_difficulty=preferred_difficulty,
                top_k=top_k,
                show_detailed=False
            )
            
            # Determine recommendation method
            recommendation_method = 'content_based'
            if recommendations and len(recommendations) > 0:
                first_rec = recommendations[0]
                if isinstance(first_rec, dict) and first_rec.get('user_type') == 'new_user_based':
                    recommendation_method = 'user_based_collaborative_filtering'
            
            # Additional data
            additional_data = {
                'user_preferences': {
                    'preferred_categories': preferred_categories,
                    'preferred_difficulty': preferred_difficulty,
                },
                'recommendation_method': recommendation_method
            }
            
            if user_info:
                additional_data['authenticated_user'] = user_info['user_id']
            
            return RecommendationSerializer.serialize_recommendations(
                recommendations=recommendations,
                user_type='new',
                additional_data=additional_data
            )
            
        except ValidationError as e:
            return RecommendationSerializer.serialize_validation_error(
                missing_fields=e.missing_fields,
                empty_fields=e.empty_fields,
                invalid_fields=e.invalid_fields
            )
        except AttributeError as e:
            logger.error(f"Recommender method not available: {str(e)}")
            return RecommendationSerializer.serialize_error(
                "Recommender method not available",
                error_code="METHOD_NOT_AVAILABLE",
                details=str(e)
            )
        except Exception as e:
            logger.error(f"Error in new user recommendations: {str(e)}")
            return RecommendationSerializer.serialize_error(
                "Failed to get recommendations",
                error_code="RECOMMENDATION_ERROR",
                details=str(e)
            )
    
    def get_content_based_recommendations(self, data, user_info=None):
        """Get content-based recommendations (fallback)"""
        try:
            validated_data = RecommendationValidator.validate_content_based_request(data, self.recommender)
            
            preferred_categories = validated_data['preferred_categories']
            top_k = validated_data['top_k']
            
            logger.info(f"Getting content-based recommendations with categories: {preferred_categories}, top_k: {top_k}")
            
            # Get recommendations from model
            recommendations = self.recommender.get_user_based_recommendations_for_new_user(
                preferred_categories=preferred_categories,
                top_k=top_k,
            )
            
            additional_data = {
                'recommendation_method': 'content_based'
            }
            
            if user_info:
                additional_data['authenticated_user'] = user_info['user_id']
            
            return RecommendationSerializer.serialize_recommendations(
                recommendations=recommendations,
                user_type='new',
                additional_data=additional_data
            )
            
        except ValidationError as e:
            return RecommendationSerializer.serialize_validation_error(
                missing_fields=e.missing_fields,
                empty_fields=e.empty_fields,
                invalid_fields=e.invalid_fields
            )
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {str(e)}")
            return RecommendationSerializer.serialize_error(
                "Failed to get content-based recommendations",
                error_code="RECOMMENDATION_ERROR",
                details=str(e)
            )
    
    def get_recommendation_options(self):
        """Get available categories and difficulty levels"""
        try:
            options = {
                'success': True,
                'categories': [],
                'difficulties': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Get available categories
            if hasattr(self.recommender, 'valid_categories'):
                if hasattr(self.recommender.valid_categories, '__iter__'):
                    options['categories'] = list(self.recommender.valid_categories)
                else:
                    options['categories'] = []
            
            # Get available difficulties
            if hasattr(self.recommender, 'difficulty_mapping'):
                options['difficulties'] = list(self.recommender.difficulty_mapping.values())
            
            return options
            
        except Exception as e:
            logger.error(f"Error getting recommendation options: {str(e)}")
            return RecommendationSerializer.serialize_error(
                "Failed to get options",
                error_code="OPTIONS_ERROR",
                details=str(e)
            )