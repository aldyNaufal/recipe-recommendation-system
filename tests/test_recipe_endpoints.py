import pytest
import json

class TestRecipeEndpoints:
    """Test suite untuk recipe detail dan search endpoints"""
    
    def test_get_recipe_detail_success(self, client):
        """Test mendapatkan detail resep - success case"""
        item_id = 1
        response = client.get(f'/api/recipe/{item_id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert data['success'] is True
        assert 'recipe' in data
        assert 'timestamp' in data
        
        # Check recipe structure
        recipe = data['recipe']
        required_fields = [
            'item_id', 'title_cleaned', 'ingredients_cleaned', 'steps_cleaned',
            'category', 'total_rating', 'total_ingredients', 'total_steps',
            'difficulty_score', 'difficulty_level', 'image_url'
        ]
        
        for field in required_fields:
            assert field in recipe
        
        # Check data types
        assert isinstance(recipe['item_id'], int)
        assert isinstance(recipe['total_rating'], float)
        assert isinstance(recipe['total_ingredients'], int)
        assert isinstance(recipe['total_steps'], int)
        assert isinstance(recipe['difficulty_score'], float)
        assert recipe['item_id'] == item_id
    
    def test_get_recipe_detail_not_found(self, client):
        """Test mendapatkan detail resep yang tidak ada"""
        item_id = 999  # ID yang tidak ada
        response = client.get(f'/api/recipe/{item_id}')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Recipe not found' in data['error']
    
    def test_get_recipe_detail_no_model(self, client_no_model):
        """Test mendapatkan detail resep tanpa model"""
        response = client_no_model.get('/api/recipe/1')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['error'] == 'Model not loaded'
    
    def test_search_recipes_success(self, client, sample_search_request):
        """Test search resep - success case"""
        response = client.post('/api/recipes/search',
                              json=sample_search_request,
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert data['success'] is True
        assert data['query'] == sample_search_request['query']
        assert 'total_results' in data
        assert 'recipes' in data
        assert 'filters_applied' in data
        assert 'timestamp' in data
        
        # Check recipes structure
        recipes = data['recipes']
        assert isinstance(recipes, list)
        
        if recipes:
            recipe = recipes[0]
            required_fields = [
                'item_id', 'title_cleaned', 'category', 'total_rating',
                'total_ingredients', 'total_steps', 'difficulty_level', 'image_url'
            ]
            
            for field in required_fields:
                assert field in recipe
    
    def test_search_recipes_missing_query(self, client):
        """Test search resep tanpa query"""
        response = client.post('/api/recipes/search',
                              json={},
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Missing required field' in data['message']
    
    def test_search_recipes_with_filters(self, client):
        """Test search resep dengan filter"""
        request_data = {
            'query': 'ayam',
            'category_filter': 'Main Course',
            'difficulty_max': 2,
            'limit': 5
        }
        
        response = client.post('/api/recipes/search',
                              json=request_data,
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check filters applied
        filters = data['filters_applied']
        assert filters['category_filter'] == 'Main Course'
        assert filters['difficulty_max'] == 2
        
        # Check result limit
        assert len(data['recipes']) <= 5
    
    def test_search_recipes_limit_max(self, client):
        """Test search resep dengan limit maksimal"""
        request_data = {
            'query': 'resep',
            'limit': 100  # Melebihi limit maksimal 50
        }
        
        response = client.post('/api/recipes/search',
                              json=request_data,
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['recipes']) <= 50
    
    def test_search_recipes_no_model(self, client_no_model, sample_search_request):
        """Test search resep tanpa model"""
        response = client_no_model.post('/api/recipes/search',
                                       json=sample_search_request,
                                       content_type='application/json')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['error'] == 'Model not loaded'