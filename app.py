from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import os
from werkzeug.exceptions import BadRequest, InternalServerError
import traceback

# Import class yang sudah Anda buat
from services.recipe_recommender import EnhancedIndonesianRecipeRecommender

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS untuk semua endpoint

# Global variables
recommender = None
MODEL_PATH = "models/fix_model.pkl"
DATA_PATH = "data/data_recipes_cleaned.csv"

def load_model():
    """Load model yang sudah ditraining"""
    global recommender
    try:
        if os.path.exists(MODEL_PATH):
            recommender = EnhancedIndonesianRecipeRecommender()
            recommender.load_model(MODEL_PATH)
            logger.info("✅ Model berhasil dimuat")
            return True
        else:
            logger.error(f"❌ Model file tidak ditemukan: {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def validate_request_data(data, required_fields):
    """Validasi data request"""
    for field in required_fields:
        if field not in data:
            raise BadRequest(f"Missing required field: {field}")

# ================================
# HEALTH CHECK & INFO ENDPOINTS
# ================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': recommender is not None
    })

@app.route('/api/info', methods=['GET'])
def get_system_info():
    """Informasi sistem rekomendasi"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'system_name': 'Indonesian Recipe Recommendation System',
            'version': '1.0.0',
            'total_recipes': len(recommender.original_data) if recommender.original_data is not None else 0,
            'total_users': len(recommender.processed_data['user_id'].unique()) if recommender.processed_data is not None else 0,
            'available_categories': recommender.valid_categories,
            'difficulty_levels': list(recommender.difficulty_mapping.values()),
            'model_type': 'Hybrid Collaborative Filtering + Content-Based',
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return jsonify({'error': 'Failed to get system info'}), 500

# ================================
# RECOMMENDATION ENDPOINTS
# ================================



# Helper function untuk convert numpy/pandas types
def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
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
    else:
        return obj

# Improved validation function
def validate_request_data(data, required_fields):
    """Validate request data with better error messages"""
    if not data:
        raise BadRequest("Request body is empty or invalid JSON")
    
    missing_fields = []
    empty_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
        elif data[field] is None or str(data[field]).strip() == '':
            empty_fields.append(field)
    
    if missing_fields:
        raise BadRequest(f"Missing required fields: {', '.join(missing_fields)}")
    
    if empty_fields:
        raise BadRequest(f"Empty fields not allowed: {', '.join(empty_fields)}")

@app.route('/api/recommendations/existing-user', methods=['POST'])
def get_existing_user_recommendations():
    """Rekomendasi untuk user yang sudah ada - POST method"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Debug logging
        logger.info(f"Request method: {request.method}")
        logger.info(f"Content-Type: {request.headers.get('Content-Type')}")
        
        # Cek Content-Type
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")
        
        # Validasi data
        validate_request_data(data, ['user_id'])
        
        user_id = data['user_id']
        top_k = data.get('top_k', 10)
        
        # Validasi dan convert types
        try:
            user_id = int(user_id)
            top_k = int(top_k)
            
            if top_k <= 0:
                top_k = 10
            elif top_k > 50:
                top_k = 50
                
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': f'Invalid parameter types: {str(e)}'
            }), 400
        
        logger.info(f"Getting recommendations for user: {user_id}, top_k: {top_k}")
        
        # Get recommendations
        recommendations = recommender.get_enhanced_recommendations(
            user_id=user_id,
            top_k=top_k,
            show_detailed=False
        )
        
        # Convert to JSON serializable format
        clean_recommendations = convert_to_serializable(recommendations)
        
        response_data = {
            'success': True,
            'user_id': user_id,
            'user_type': 'existing', 
            'total_recommendations': len(clean_recommendations),
            'recommendations': clean_recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except BadRequest as e:
        logger.error(f"Bad Request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except ValueError as e:
        logger.error(f"Value Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Invalid parameter value: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Error in existing user recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Failed to get recommendations',
            'details': str(e) if app.debug else 'Internal server error'
        }), 500


@app.route('/api/recommendations/existing-user/<int:user_id>', methods=['GET'])
def get_existing_user_recommendations_get(user_id):
    """Rekomendasi untuk user yang sudah ada - GET method"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        top_k = request.args.get('top_k', 10, type=int)
        
        # Validasi parameter
        if top_k <= 0:
            top_k = 10
        elif top_k > 50:
            top_k = 50
        
        logger.info(f"Getting recommendations for user: {user_id}, top_k: {top_k}")
        
        # Get recommendations
        recommendations = recommender.get_enhanced_recommendations(
            user_id=user_id,
            top_k=top_k,
            show_detailed=False
        )
        
        # Convert to JSON serializable format
        clean_recommendations = convert_to_serializable(recommendations)
        
        response_data = {
            'success': True,
            'user_id': user_id,
            'user_type': 'existing',
            'total_recommendations': len(clean_recommendations),
            'recommendations': clean_recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in existing user recommendations GET: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Failed to get recommendations',
            'details': str(e) if app.debug else 'Internal server error'
        }), 500


# Test endpoint untuk debugging
@app.route('/api/test/json-serialize', methods=['POST'])
def test_json_serialize():
    """Test endpoint untuk debugging JSON serialization"""
    try:
        data = request.get_json()
        
        # Simulasi data dengan numpy types
        test_data = {
            'numpy_int': np.int64(123),
            'numpy_float': np.float64(45.67),
            'numpy_array': np.array([1, 2, 3]),
            'regular_data': {'name': 'test', 'value': 100},
            'input_data': data
        }
        
        # Convert and return
        clean_data = convert_to_serializable(test_data)
        
        return jsonify({
            'success': True,
            'original_types': str(type(test_data)),
            'converted_data': clean_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/recommendations/new-user', methods=['POST'])
def get_new_user_recommendations():
    """Rekomendasi untuk user baru berdasarkan preferensi"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Debug logging
        logger.info(f"Request method: {request.method}")
        logger.info(f"Content-Type: {request.headers.get('Content-Type')}")
        
        # Cek Content-Type
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")
        
        # Validasi data tidak kosong
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is empty or invalid JSON'
            }), 400
        
        # Validasi required fields
        if 'preferred_categories' not in data:
            return jsonify({
                'success': False,
                'error': 'preferred_categories is required'
            }), 400
        
        preferred_categories = data['preferred_categories']
        
        # Validasi preferred_categories tidak kosong
        if not preferred_categories or not isinstance(preferred_categories, list):
            return jsonify({
                'success': False,
                'error': 'preferred_categories must be a non-empty list'
            }), 400
        
        preferred_difficulty = data.get('preferred_difficulty')
        top_k = data.get('top_k', 10)
        
        # Validasi dan convert top_k
        try:
            top_k = int(top_k)
            if top_k <= 0:
                top_k = 10
            elif top_k > 50:
                top_k = 50
        except (ValueError, TypeError):
            top_k = 10
        
        # Validasi kategori
        valid_categories = []
        if hasattr(recommender, 'valid_categories'):
            valid_categories = [cat for cat in preferred_categories if cat in recommender.valid_categories]
            
            if not valid_categories:
                return jsonify({
                    'success': False,
                    'error': 'No valid categories provided',
                    'provided_categories': preferred_categories,
                    'available_categories': list(recommender.valid_categories) if hasattr(recommender.valid_categories, '__iter__') else []
                }), 400
        else:
            # Jika tidak ada valid_categories, gunakan semua yang diberikan
            valid_categories = preferred_categories
            logger.warning("Recommender doesn't have valid_categories attribute, using all provided categories")
        
        # Validasi difficulty
        if preferred_difficulty:
            if hasattr(recommender, 'difficulty_mapping'):
                if preferred_difficulty not in recommender.difficulty_mapping.values():
                    return jsonify({
                        'success': False,
                        'error': 'Invalid difficulty level',
                        'provided_difficulty': preferred_difficulty,
                        'available_difficulties': list(recommender.difficulty_mapping.values())
                    }), 400
            else:
                logger.warning("Recommender doesn't have difficulty_mapping attribute")
        
        logger.info(f"Getting new user recommendations with categories: {valid_categories}, difficulty: {preferred_difficulty}, top_k: {top_k}")
        
        # Get recommendations
        recommendations = recommender.get_user_profile_based_recommendations(
            preferred_categories=valid_categories,
            preferred_difficulty=preferred_difficulty,
            top_k=top_k,
            show_detailed=False
        )
        
        # Convert to JSON serializable format
        clean_recommendations = convert_to_serializable(recommendations)
        
        # Determine recommendation method
        recommendation_method = 'content_based'  # default
        if clean_recommendations and len(clean_recommendations) > 0:
            first_rec = clean_recommendations[0]
            if isinstance(first_rec, dict) and first_rec.get('user_type') == 'new_user_based':
                recommendation_method = 'user_based_collaborative_filtering'
        
        response_data = {
            'success': True,
            'user_type': 'new',
            'total_recommendations': len(clean_recommendations),
            'recommendations': clean_recommendations,
            'user_preferences': {
                'preferred_categories': valid_categories,
                'preferred_difficulty': preferred_difficulty,
            },
            'recommendation_method': recommendation_method,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except BadRequest as e:
        logger.error(f"Bad Request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except ValueError as e:
        logger.error(f"Value Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Invalid parameter value: {str(e)}'
        }), 400
    except AttributeError as e:
        logger.error(f"Attribute Error (recommender method): {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Recommender method not available',
            'details': str(e) if app.debug else 'Method not implemented'
        }), 500
    except Exception as e:
        logger.error(f"Error in new user recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Failed to get recommendations',
            'details': str(e) if app.debug else 'Internal server error'
        }), 500


# Alternative GET method untuk new user dengan query parameters
@app.route('/api/recommendations/new-user-get', methods=['GET'])
def get_new_user_recommendations_get():
    """Rekomendasi untuk user baru - GET method dengan query parameters"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get categories from query params (comma-separated)
        categories_param = request.args.get('categories', '')
        if not categories_param:
            return jsonify({
                'success': False,
                'error': 'categories parameter is required (comma-separated list)'
            }), 400
        
        preferred_categories = [cat.strip() for cat in categories_param.split(',') if cat.strip()]
        
        if not preferred_categories:
            return jsonify({
                'success': False,
                'error': 'At least one category must be provided'
            }), 400
        
        preferred_difficulty = request.args.get('difficulty')
        top_k = request.args.get('top_k', 10, type=int)
        
        # Validasi parameter
        if top_k <= 0:
            top_k = 10
        elif top_k > 50:
            top_k = 50
        
        # Validasi kategori
        valid_categories = []
        if hasattr(recommender, 'valid_categories'):
            valid_categories = [cat for cat in preferred_categories if cat in recommender.valid_categories]
            
            if not valid_categories:
                return jsonify({
                    'success': False,
                    'error': 'No valid categories provided',
                    'provided_categories': preferred_categories,
                    'available_categories': list(recommender.valid_categories) if hasattr(recommender.valid_categories, '__iter__') else []
                }), 400
        else:
            valid_categories = preferred_categories
        
        logger.info(f"Getting new user recommendations (GET) with categories: {valid_categories}, difficulty: {preferred_difficulty}, top_k: {top_k}")
        
        # Get recommendations
        recommendations = recommender.get_user_profile_based_recommendations(
            preferred_categories=valid_categories,
            preferred_difficulty=preferred_difficulty,
            top_k=top_k,
            show_detailed=False
        )
        
        # Convert to JSON serializable format
        clean_recommendations = convert_to_serializable(recommendations)
        
        response_data = {
            'success': True,
            'user_type': 'new',
            'total_recommendations': len(clean_recommendations),
            'recommendations': clean_recommendations,
            'user_preferences': {
                'preferred_categories': valid_categories,
                'preferred_difficulty': preferred_difficulty,
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in new user recommendations GET: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Failed to get recommendations',
            'details': str(e) if app.debug else 'Internal server error'
        }), 500


# Endpoint untuk mendapatkan kategori dan difficulty yang tersedia
@app.route('/api/recommendations/options', methods=['GET'])
def get_recommendation_options():
    """Get available categories and difficulty levels"""
    try:
        options = {
            'success': True,
            'categories': [],
            'difficulties': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Get available categories
        if hasattr(recommender, 'valid_categories'):
            if hasattr(recommender.valid_categories, '__iter__'):
                options['categories'] = list(recommender.valid_categories)
            else:
                options['categories'] = []
        
        # Get available difficulties
        if hasattr(recommender, 'difficulty_mapping'):
            options['difficulties'] = list(recommender.difficulty_mapping.values())
        
        return jsonify(options)
        
    except Exception as e:
        logger.error(f"Error getting recommendation options: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get options',
            'details': str(e) if app.debug else 'Internal server error'
        }), 500


# Test endpoint untuk new user recommendations
@app.route('/api/test/new-user-sample', methods=['GET'])
def test_new_user_sample():
    """Test endpoint dengan sample data untuk new user"""
    sample_data = {
        'preferred_categories': ['Indonesian', 'Asian', 'Vegetarian'],
        'preferred_difficulty': 'Easy',
        'top_k': 5
    }
    
    # Simulate POST request
    try:
        if not recommender:
            return jsonify({'error': 'Model not loaded'}), 500
        
        recommendations = recommender.get_user_profile_based_recommendations(
            preferred_categories=sample_data['preferred_categories'],
            preferred_difficulty=sample_data['preferred_difficulty'],
            top_k=sample_data['top_k'],
            show_detailed=False
        )
        
        clean_recommendations = convert_to_serializable(recommendations)
        
        return jsonify({
            'success': True,
            'message': 'Sample new user recommendations',
            'sample_input': sample_data,
            'total_recommendations': len(clean_recommendations),
            'recommendations': clean_recommendations[:3],  # Show only first 3
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/recommendations/content-based', methods=['POST'])
def get_content_based_recommendations():
    """Rekomendasi berbasis konten (fallback)"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        preferred_categories = data.get('preferred_categories')
        top_k = data.get('top_k', 10)
        
        if top_k > 50:
            top_k = 50
        
        recommendations = recommender.get_user_based_recommendations_for_new_user(
            preferred_categories=preferred_categories,
            top_k=top_k,
        )
        
        # Convert to serializable format (same as other endpoints)
        clean_recommendations = convert_to_serializable(recommendations)
        
        return jsonify({
            'success': True,
            'user_type': 'new',
            'recommendation_method': 'content_based',
            'total_recommendations': len(clean_recommendations),
            'recommendations': clean_recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in content-based recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get content-based recommendations',
            'details': str(e) if app.debug else 'Internal server error'
        }), 500
    
# ================================
# RECIPE DETAIL ENDPOINTS
# ================================

@app.route('/api/recipe/<int:item_id>', methods=['GET'])
def get_recipe_detail(item_id):
    """Detail resep berdasarkan item_id"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        recipe_data = recommender.original_data[recommender.original_data['item_id'] == item_id]
        
        if recipe_data.empty:
            return jsonify({
                'success': False,
                'error': 'Recipe not found'
            }), 404
        
        recipe = recipe_data.iloc[0]
        
        # Gunakan method dari recommender class
        difficulty_score = recommender.calculate_difficulty_score(
            int(recipe['Total Ingredients']), 
            int(recipe['Total Steps'])
        )
        
        recipe_detail = {
            'item_id': int(recipe['item_id']),
            'title_cleaned': recipe['Title Cleaned'],
            'ingredients_cleaned': recipe['Ingredients Cleaned'],
            'steps_cleaned': recipe['Steps Cleaned'],
            'category': recipe.get('Category', 'Unknown'),
            'total_rating': float(recipe['total_rating']),
            'total_ingredients': int(recipe['Total Ingredients']),
            'total_steps': int(recipe['Total Steps']),
            'difficulty_score': float(difficulty_score),
            'difficulty_level': recommender.get_difficulty_level(difficulty_score),
            'image_url': recipe.get('Image URL', 'N/A')
        }
        
        return jsonify({
            'success': True,
            'recipe': recipe_detail,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting recipe detail: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get recipe detail',
            'details': str(e)
        }), 500

# Simplified Recipe Search Endpoint - hanya ingredients dan limit
@app.route('/api/recipes/search', methods=['POST'])
def search_recipes():
    """
    Pencarian resep berdasarkan ingredients yang dimiliki user
    Input sederhana: ingredients (required) dan limit (optional)
    """
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Debug log
        logger.info(f"Received search request: {data}")
        
        # Validation untuk ingredients - coba beberapa format
        ingredients_input = None
        possible_fields = ['ingredients', 'ingredients_cleaned', 'Ingredients Cleaned']
        
        for field in possible_fields:
            if field in data and data[field]:
                ingredients_input = data[field]
                break
        
        if ingredients_input is None:
            return jsonify({
                'success': False,
                'message': 'Missing required field: ingredients',
                'example': {
                    'ingredients': ['ayam', 'bawang merah', 'tomat'],
                    'limit': 20
                }
            }), 400
        
        # Convert string ke list jika perlu
        if isinstance(ingredients_input, str):
            ingredients_input = [ing.strip() for ing in ingredients_input.split(',') if ing.strip()]
        
        if not isinstance(ingredients_input, list) or len(ingredients_input) == 0:
            return jsonify({
                'success': False,
                'message': 'ingredients must be a non-empty list or comma-separated string',
                'example': ['ayam', 'bawang merah', 'tomat']
            }), 400
        
        # Validation untuk limit
        limit = data.get('limit', 20)  # Default 20
        try:
            limit = int(limit)
            limit = min(max(limit, 1), 50)  # Between 1-50
        except (ValueError, TypeError):
            limit = 20
        
        logger.info(f"Searching recipes with ingredients: {ingredients_input}, limit: {limit}")
        
        # Call the search function dengan parameter default yang masuk akal
        search_results = recommender.search_recipes_by_ingredients(
            ingredients_input=ingredients_input,
            category_filter=None,           # No category filter
            difficulty_max=5,               # Allow all difficulty levels
            limit=limit,
            search_mode='any',              # Flexible matching
            min_match_percentage=0.3,       # Default threshold
            prefer_more_matches=True,       # Prefer recipes using more ingredients
            show_detailed=False             # Keep response clean
        )
        
        # Clean up results for API response
        clean_results = convert_to_serializable(search_results)
        
        # Add metadata
        clean_results.update({
            'timestamp': datetime.now().isoformat(),
            'request_info': {
                'ingredients_count': len(ingredients_input),
                'requested_limit': limit,
                'actual_results': len(clean_results.get('recipes', []))
            }
        })
        
        logger.info(f"Found {len(clean_results.get('recipes', []))} recipes")
        
        return jsonify(clean_results)
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({
            'success': False,
            'message': f'Validation error: {str(ve)}'
        }), 400
        
    except Exception as e:
        logger.error(f"Error in recipe search: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': 'Failed to search recipes',
            'error': str(e)
        }), 500
    
# ================================
# UTILITY ENDPOINTS
# ================================

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Daftar kategori yang tersedia"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'success': True,
        'categories': recommender.valid_categories,
        'total_categories': len(recommender.valid_categories)
    })

@app.route('/api/difficulties', methods=['GET'])
def get_difficulty_levels():
    """Daftar tingkat kesulitan"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'success': True,
        'difficulty_levels': list(recommender.difficulty_mapping.values()),
        'difficulty_mapping': recommender.difficulty_mapping
    })

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Statistik dataset"""
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        df = recommender.original_data
        
        stats = {
            'total_recipes': len(df),
            'total_users': len(recommender.processed_data['user_id'].unique()),
            'total_ratings': len(recommender.processed_data),
            'category_distribution': df['Category'].value_counts().to_dict(),
            'difficulty_distribution': {
                'Cepat & Mudah': len(df[df['Difficulty_Score'] <= 1.5]),
                'Butuh Usaha': len(df[(df['Difficulty_Score'] > 1.5) & (df['Difficulty_Score'] <= 2.5)]),
                'Level Dewa Masak': len(df[df['Difficulty_Score'] > 2.5])
            },
            'rating_stats': {
                'avg_rating': float(df['total_rating'].mean()),
                'min_rating': float(df['total_rating'].min()),
                'max_rating': float(df['total_rating'].max())
            }
        }
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get statistics'
        }), 500

# ================================
# ERROR HANDLERS
# ================================

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    """Handle BadRequest exceptions dari werkzeug"""
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'message': str(e.description)
    }), 400

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'message': str(error.description) if hasattr(error, 'description') else 'Bad Request'
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    # Log the actual error for debugging
    app.logger.error(f'Server Error: {error}', exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'Something went wrong on the server'
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    # Log the exception
    app.logger.error(f'Unhandled Exception: {str(e)}', exc_info=True)
    
    # Return 500 error
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500

# ================================
# MAIN
# ================================

if __name__ == '__main__':
    # Load model saat startup
    if not load_model():
        logger.error("❌ Gagal memuat model. Pastikan model sudah ditraining dan tersimpan.")
        exit(1)
    
    # Jalankan Flask app
    logger.info("🚀 Starting Recipe Recommendation API...")
    app.run(
        host='0.0.0.0',  # Agar bisa diakses dari luar
        port=5000,
        debug=False,  # Set False untuk production
        threaded=True
    )