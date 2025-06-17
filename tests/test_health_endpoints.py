import pytest
import json
from datetime import datetime

class TestHealthEndpoints:
    """Test suite untuk health check dan info endpoints"""
    
    def test_health_check_with_model(self, client, mock_recipe_data):
        """Test health check ketika model sudah loaded"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['model_loaded'] is True
        
        # Validate timestamp format
        timestamp = datetime.fromisoformat(data['timestamp'])
        assert isinstance(timestamp, datetime)
    
    def test_health_check_without_model(self, client_no_model):
        """Test health check ketika model belum loaded"""
        response = client_no_model.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is False
    
    def test_system_info_with_model(self, client, mock_recipe_data):
        """Test system info ketika model sudah loaded"""
        response = client.get('/api/info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check required fields
        required_fields = [
            'system_name', 'version', 'total_recipes', 'total_users',
            'available_categories', 'difficulty_levels', 'model_type', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in data
        
        # Validate data types
        assert isinstance(data['total_recipes'], int)
        assert isinstance(data['total_users'], int)
        assert isinstance(data['available_categories'], list)
        assert isinstance(data['difficulty_levels'], list)
        assert data['system_name'] == 'Indonesian Recipe Recommendation System'
        assert data['version'] == '1.0.0'
        assert data['model_type'] == 'Hybrid Collaborative Filtering + Content-Based'
    
    def test_system_info_without_model(self, client_no_model):
        """Test system info ketika model belum loaded"""
        response = client_no_model.get('/api/info')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Model not loaded'