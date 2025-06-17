from flask import Blueprint
from controllers.recommendation_controller import RecommendationController
from services.auth_service import require_auth, optional_auth
import logging

logger = logging.getLogger(__name__)

# Create blueprint
recommendation_bp = Blueprint('recommendations', __name__, url_prefix='/api/recommendations')

# Initialize controller (will be set by app factory)
recommendation_controller = None

def init_routes(recommender):
    """Initialize routes with recommender"""
    global recommendation_controller
    recommendation_controller = RecommendationController(recommender)
    logger.info("Recommendation routes initialized")

# ================================
# AUTHENTICATED ENDPOINTS
# ================================

@recommendation_bp.route('/existing-user', methods=['POST'])
@require_auth
def existing_user_recommendations_post():
    """Rekomendasi untuk user yang sudah ada - POST method (requires auth)"""
    return recommendation_controller.get_existing_user_recommendations_post()

@recommendation_bp.route('/existing-user/<int:user_id>', methods=['GET'])
@require_auth
def existing_user_recommendations_get(user_id):
    """Rekomendasi untuk user yang sudah ada - GET method (requires auth)"""
    return recommendation_controller.get_existing_user_recommendations_get(user_id)

@recommendation_bp.route('/new-user', methods=['POST'])
@require_auth
def new_user_recommendations_post():
    """Rekomendasi untuk user baru berdasarkan preferensi - POST (requires auth)"""
    return recommendation_controller.get_new_user_recommendations_post()

@recommendation_bp.route('/new-user-get', methods=['GET'])
@require_auth
def new_user_recommendations_get():
    """Rekomendasi untuk user baru - GET method dengan query parameters (requires auth)"""
    return recommendation_controller.get_new_user_recommendations_get()

@recommendation_bp.route('/content-based', methods=['POST'])
@require_auth
def content_based_recommendations_post():
    """Rekomendasi berbasis konten (fallback) (requires auth)"""
    return recommendation_controller.get_content_based_recommendations_post()

@recommendation_bp.route('/options', methods=['GET'])
@require_auth
def recommendation_options():
    """Get available categories and difficulty levels (requires auth)"""
    return recommendation_controller.get_recommendation_options()

# ================================
# TEST/DEBUG ENDPOINTS (Optional Auth)
# ================================

@recommendation_bp.route('/test/new-user-sample', methods=['GET'])
@optional_auth
def test_new_user_sample():
    """Test endpoint dengan sample data untuk new user (optional auth)"""
    return recommendation_controller.test_new_user_sample()

# ================================
# HEALTH CHECK ENDPOINT (No Auth)
# ================================

@recommendation_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = "healthy" if recommendation_controller and recommendation_controller.recommendation_service.recommender else "unhealthy"
        return {
            'success': True,
            'status': status,
            'service': 'recommendation_service',
            'message': 'Recommendation service is running'
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            'success': False,
            'status': 'error',
            'service': 'recommendation_service',
            'message': str(e)
        }, 500

# ================================
# ERROR HANDLERS
# ================================

@recommendation_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return {
        'success': False,
        'error': 'Endpoint not found',
        'code': 'ENDPOINT_NOT_FOUND'
    }, 404

@recommendation_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return {
        'success': False,
        'error': 'Method not allowed',
        'code': 'METHOD_NOT_ALLOWED'
    }, 405

@recommendation_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return {
        'success': False,
        'error': 'Internal server error',
        'code': 'INTERNAL_ERROR'
    }, 500