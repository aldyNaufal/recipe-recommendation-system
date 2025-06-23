import os
from flask import jsonify
from core.logger import AppLogger

class UtilityRoutes:
    """Utility routes like health check, debug, API info"""
    
    def __init__(self):
        self.logger = AppLogger.get_logger(__name__)
    
    def register_routes(self, app, config_name, recommender):
        """Register utility routes with Flask app"""
        
        @app.route('/api/recommendations/health', methods=['GET'])
        def health_check():
            """Application health check"""
            try:
                status = self._get_health_status(config_name, recommender)
                return jsonify(status)
            except Exception as e:
                self.logger.error(f"Health check error: {str(e)}")
                return jsonify({
                    'success': False,
                    'status': 'error',
                    'service': 'recommendation_api',
                    'message': str(e)
                }), 500
        
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
        
        # Debug endpoint (development only)
        if config_name == 'development':
            @app.route('/debug/model', methods=['GET'])
            def debug_model():
                """Debug model loading status"""
                try:
                    debug_info = self._get_debug_info(recommender)
                    return jsonify(debug_info)
                except Exception as e:
                    self.logger.error(f"Debug endpoint error: {str(e)}")
                    return jsonify({'error': str(e)}), 500
    
    def _get_health_status(self, config_name, recommender):
        """Get application health status"""
        status = {
            'success': True,
            'status': 'healthy',
            'service': 'recommendation_api',
            'version': '1.0.0',
            'environment': config_name,
            'recommender_loaded': recommender is not None
        }
        
        if recommender is not None:
            # PERBAIKAN: Gunakan method yang benar
            try:
                # Cek apakah ada method is_loaded
                if hasattr(recommender, 'is_loaded') and callable(recommender.is_loaded):
                    model_status = recommender.is_loaded()
                    status.update({
                        'model_loaded': model_status.get('model_loaded', False),
                        'components_status': model_status.get('components', {}),
                        'is_ready': recommender.is_model_loaded() if hasattr(recommender, 'is_model_loaded') else False
                    })
                else:
                    # Fallback: gunakan is_model_loaded langsung
                    is_ready = recommender.is_model_loaded() if hasattr(recommender, 'is_model_loaded') else False
                    status.update({
                        'model_loaded': is_ready,
                        'components_status': {
                            'model': hasattr(recommender, 'model') and recommender.model is not None,
                            'encoders': hasattr(recommender, 'encoders') and recommender.encoders is not None,
                            'processed_data': hasattr(recommender, 'processed_data') and recommender.processed_data is not None
                        },
                        'is_ready': is_ready
                    })
            except Exception as e:
                self.logger.error(f"Error checking model status: {str(e)}")
                status.update({
                    'model_loaded': False,
                    'components_status': {'error': str(e)},
                    'is_ready': False
                })
        else:
            status.update({
                'model_loaded': False,
                'components_status': {},
                'is_ready': False
            })
        
        return status
    
    def _get_debug_info(self, recommender):
        """Get debug information about model status"""
        debug_info = {
            'current_directory': os.getcwd(),
            'recommender_exists': recommender is not None,
            'model_files_check': []
        }
        
        models_dir = os.path.join(os.getcwd(), 'models')
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            debug_info['models_directory_files'] = files
            
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
        
        # PERBAIKAN: Ganti get_model_status() dengan method yang benar
        if recommender is not None:
            try:
                # Coba berbagai method yang mungkin ada
                if hasattr(recommender, 'is_loaded') and callable(recommender.is_loaded):
                    debug_info['model_status'] = recommender.is_loaded()
                elif hasattr(recommender, 'is_model_loaded') and callable(recommender.is_model_loaded):
                    debug_info['model_status'] = {
                        'is_loaded': recommender.is_model_loaded(),
                        'components': {
                            'model': hasattr(recommender, 'model') and recommender.model is not None,
                            'encoders': hasattr(recommender, 'encoders') and recommender.encoders is not None,
                            'processed_data': hasattr(recommender, 'processed_data') and recommender.processed_data is not None
                        }
                    }
                else:
                    # Manual check jika tidak ada method
                    debug_info['model_status'] = {
                        'available_methods': [method for method in dir(recommender) if not method.startswith('_')],
                        'manual_check': {
                            'has_model': hasattr(recommender, 'model'),
                            'has_encoders': hasattr(recommender, 'encoders'),
                            'has_processed_data': hasattr(recommender, 'processed_data'),
                            'model_is_none': recommender.model is None if hasattr(recommender, 'model') else 'N/A',
                            'encoders_is_none': recommender.encoders is None if hasattr(recommender, 'encoders') else 'N/A',
                            'processed_data_is_none': recommender.processed_data is None if hasattr(recommender, 'processed_data') else 'N/A'
                        }
                    }
            except Exception as e:
                debug_info['model_status'] = {
                    'error': str(e),
                    'recommender_type': type(recommender).__name__
                }
        
        return debug_info