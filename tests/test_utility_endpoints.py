import pytest
import json

class TestUtilityEndpoints:
    """Test suite untuk utility endpoints"""
    
    def test_get_categories_success(self, client):
        """Test mendapatkan daftar kategori - success case"""
        response = client.get('/api/categories')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert data['success'] is True
        assert 'categories' in data
        assert 'total_categories' in data
        
        # Check categories
        categories = data['categories']
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert data['total_categories'] == len(categories)
        
        # Check if expected categories exist
        expected_categories = ['Main Course', 'Appetizer', 'Soup', 'Dessert']
        for cat in expected_categories:
            assert cat in categories
    
    def test_get_categories_no_model(self, client_no_model):
        """Test mendapatkan kategori tanpa model"""
        response = client_no_model.get('/api/categories')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['error'] == 'Model not loaded'
    
    def test_get_difficulty_levels_success(self, client):
        """Test mendapatkan tingkat kesulitan - success case"""
        response = client.get('/api/difficulties')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert data['success'] is True
        assert 'difficulty_levels' in data
        assert 'difficulty_mapping' in data
        
        # Check difficulty levels
        levels = data['difficulty_levels']
        assert isinstance(levels, list)
        assert len(levels) > 0
        
        # Check if expected difficulty levels exist
        expected_levels = ['Cepat & Mudah', 'Butuh Usaha', 'Level Dewa Masak']
        for level in expected_levels:
            assert level in levels
        
        # Check difficulty mapping
        mapping = data['difficulty_mapping']
        assert isinstance(mapping, dict)
        assert len(mapping) == 3
    
    def test_get_difficulty_levels_no_model(self, client_no_model):
        """Test mendapatkan difficulty levels tanpa model"""
        response = client_no_model.get('/api/difficulties')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['error'] == 'Model not loaded'
    
    def test_get_statistics_success(self, client):
        """Test mendapatkan statistik - success case"""
        response = client.get('/api/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert data['success'] is True
        assert 'statistics' in data
        assert 'timestamp' in data
        
        # Check statistics structure
        stats = data['statistics']
        required_fields = [
            'total_recipes', 'total_users', 'total_ratings',
            'category_distribution', 'difficulty_distribution', 'rating_stats'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Check data types
        assert isinstance(stats['total_recipes'], int)
        assert isinstance(stats['total_users'], int)
        assert isinstance(stats['total_ratings'], int)
        assert isinstance(stats['category_distribution'], dict)
        assert isinstance(stats['difficulty_distribution'], dict)
        assert isinstance(stats['rating_stats'], dict)
        
        # Check rating stats structure
        rating_stats = stats['rating_stats']
        assert 'avg_rating' in rating_stats
        assert 'min_rating' in rating_stats
        assert 'max_rating' in rating_stats
        
        # Check difficulty distribution
        diff_dist = stats['difficulty_distribution']
        expected_difficulties = ['Cepat & Mudah', 'Butuh Usaha', 'Level Dewa Masak']
        for diff in expected_difficulties:
            assert diff in diff_dist
            assert isinstance(diff_dist[diff], int)
    
    def test_get_statistics_no_model(self, client_no_model):
        """Test mendapatkan statistik tanpa model"""
        response = client_no_model.get('/api/stats')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['error'] == 'Model not loaded'