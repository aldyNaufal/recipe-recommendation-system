import pytest
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
from unittest.mock import Mock, MagicMock
import sys
import os
import pytest
import pandas as pd

# Tambahkan root directory ke sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
        
# Mock data untuk testing
@pytest.fixture
def mock_recipe_data():
    """Mock data resep untuk testing"""
    return pd.DataFrame({
        'item_id': [1, 2, 3, 4, 5],
        'Title Cleaned': [
            'Nasi Goreng Spesial',
            'Rendang Daging Sapi',
            'Gado-gado Jakarta',
            'Ayam Bakar Bumbu Rujak',
            'Soto Ayam Lamongan'
        ],
        'Ingredients Cleaned': [
            'nasi putih, telur, kecap manis, bawang merah, cabai',
            'daging sapi, santan, cabai merah, lengkuas, kunyit',
            'tahu, tempe, kangkung, kacang tanah, petis',
            'ayam, cabai rawit, tomat, bawang putih, kecap',
            'ayam, kunyit, jahe, serai, daun jeruk'
        ],
        'Steps Cleaned': [
            'tumis bumbu, masukkan nasi, aduk rata',
            'haluskan bumbu, masak dengan santan hingga empuk',
            'rebus sayuran, siapkan bumbu kacang, campur',
            'lumuri ayam dengan bumbu, bakar hingga matang',
            'rebus ayam dengan rempah, sajikan dengan pelengkap'
        ],
        'Category': ['Main Course', 'Main Course', 'Appetizer', 'Main Course', 'Soup'],
        'total_rating': [4.5, 4.8, 4.2, 4.3, 4.6],
        'Total Ingredients': [8, 15, 12, 10, 13],
        'Total Steps': [5, 8, 6, 4, 7],
        'Difficulty_Score': [1.2, 2.8, 1.8, 1.5, 2.1],
        'Image URL': [
            'https://example.com/nasi-goreng.jpg',
            'https://example.com/rendang.jpg',
            'https://example.com/gado-gado.jpg',
            'https://example.com/ayam-bakar.jpg',
            'https://example.com/soto-ayam.jpg'
        ]
    })

@pytest.fixture
def mock_processed_data():
    """Mock processed data untuk testing"""
    return pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3, 4, 5],
        'item_id': [1, 2, 1, 3, 2, 4, 5, 1],
        'rating': [4.5, 4.8, 4.0, 4.2, 5.0, 4.3, 4.6, 3.8]
    })

@pytest.fixture
def mock_recommender(mock_recipe_data, mock_processed_data):
    """Mock recommender object - PERBAIKAN: inject fixtures sebagai parameter"""
    recommender = MagicMock()
    
    # Mock attributes - gunakan parameter fixture, bukan pemanggilan langsung
    recommender.original_data = mock_recipe_data
    recommender.processed_data = mock_processed_data
    recommender.valid_categories = ['Main Course', 'Appetizer', 'Soup', 'Dessert']
    recommender.difficulty_mapping = {
        1: 'Cepat & Mudah',
        2: 'Butuh Usaha', 
        3: 'Level Dewa Masak'
    }
    
    # Mock methods
    def mock_get_enhanced_recommendations(user_id, top_k=10, category_filter=None, 
                                        difficulty_max=3, min_rating=3.0, show_detailed=False):
        return [
            {
                'item_id': 1,
                'title_cleaned': 'Nasi Goreng Spesial',
                'predicted_rating': 4.5,
                'category': 'Main Course',
                'difficulty_level': 'Cepat & Mudah'
            },
            {
                'item_id': 2,
                'title_cleaned': 'Rendang Daging Sapi',
                'predicted_rating': 4.8,
                'category': 'Main Course',
                'difficulty_level': 'Level Dewa Masak'
            }
        ]
    
    def mock_get_user_profile_based_recommendations(preferred_categories, preferred_difficulty=None,
                                                  top_k=10, min_rating=3.0, show_detailed=False):
        return [
            {
                'item_id': 3,
                'title_cleaned': 'Gado-gado Jakarta',
                'predicted_rating': 4.2,
                'category': 'Appetizer',
                'difficulty_level': 'Butuh Usaha',
                'user_type': 'new_user_based'
            }
        ]
    
    def mock_content_based_recommendations(category_filter=None, difficulty_max=3,
                                         top_k=10, min_rating=3.0, show_detailed=False):
        return [
            {
                'item_id': 4,
                'title_cleaned': 'Ayam Bakar Bumbu Rujak',
                'predicted_rating': 4.3,
                'category': 'Main Course',
                'difficulty_level': 'Cepat & Mudah'
            }
        ]
    
    def mock_calculate_difficulty_level(score):
        if score <= 1.5:
            return 'Cepat & Mudah'
        elif score <= 2.5:
            return 'Butuh Usaha'
        else:
            return 'Level Dewa Masak'
    
    recommender.get_enhanced_recommendations = mock_get_enhanced_recommendations
    recommender.get_user_profile_based_recommendations = mock_get_user_profile_based_recommendations
    recommender._get_content_based_recommendations_for_new_user = mock_content_based_recommendations
    recommender._calculate_difficulty_level = mock_calculate_difficulty_level
    
    return recommender

@pytest.fixture
def client(mock_recommender):
    """Flask test client dengan mock recommender"""
    # Patch global recommender variable
    import app
    app.recommender = mock_recommender
    
    app.app.config['TESTING'] = True
    app.app.config['WTF_CSRF_ENABLED'] = False
    
    with app.app.test_client() as client:
        with app.app.app_context():
            yield client

@pytest.fixture
def client_no_model():
    """Flask test client tanpa model (untuk testing error cases)"""
    import app
    app.recommender = None
    
    app.app.config['TESTING'] = True
    app.app.config['WTF_CSRF_ENABLED'] = False
    
    with app.app.test_client() as client:
        with app.app.app_context():
            yield client

@pytest.fixture
def sample_user_request():
    """Sample request data untuk existing user"""
    return {
        'user_id': 1,
        'top_k': 10,
    }

@pytest.fixture
def sample_new_user_request():
    """Sample request data untuk new user"""
    return {
        'preferred_categories': ['Main Course', 'Appetizer'],
        'preferred_difficulty': 'Cepat & Mudah',
        'top_k': 5,
    }

@pytest.fixture
def sample_search_request():
    """Sample request data untuk search"""
    return {
        'query': 'ayam',
        'category_filter': 'Main Course',
        'difficulty_max': 2,
        'limit': 10
    }