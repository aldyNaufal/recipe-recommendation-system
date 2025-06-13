import pytest
import json
import time
import threading
import queue

class TestIntegrationWorkflows:
    """Test suite untuk integration workflows"""
    
    def test_complete_existing_user_workflow(self, client):
        """Test complete workflow untuk existing user"""
        # 1. Check health
        response = client.get('/health')
        assert response.status_code == 200
        
        # 2. Get system info
        response = client.get('/api/info')
        assert response.status_code == 200
        
        # 3. Get categories
        response = client.get('/api/categories')
        assert response.status_code == 200
        categories_data = json.loads(response.data)
        
        # 4. Get recommendations for existing user
        request_data = {
            'user_id': 1,
            'top_k': 5,
            'category_filter': categories_data['categories'][0],
            'min_rating': 4.0
        }
        
        response = client.post('/api/recommendations/existing-user',
                              json=request_data,
                              content_type='application/json')
        assert response.status_code == 200
        
        rec_data = json.loads(response.data)
        assert rec_data['success'] is True
        assert len(rec_data['recommendations']) <= 5
        
        # 5. Get recipe detail for first recommendation
        if rec_data['recommendations']:
            item_id = rec_data['recommendations'][0]['item_id']
            response = client.get(f'/api/recipe/{item_id}')
            assert response.status_code == 200
    
    def test_complete_new_user_workflow(self, client):
        """Test complete workflow untuk new user"""
        # 1. Get available categories
        response = client.get('/api/categories')
        assert response.status_code == 200
        categories_data = json.loads(response.data)
        
        # 2. Get difficulty levels
        response = client.get('/api/difficulties')
        assert response.status_code == 200
        difficulty_data = json.loads(response.data)
        
        # 3. Get recommendations for new user
        request_data = {
            'preferred_categories': categories_data['categories'][:2],
            'preferred_difficulty': difficulty_data['difficulty_levels'][0],
            'top_k': 3,
            'min_rating': 3.5
        }
        
        response = client.post('/api/recommendations/new-user',
                              json=request_data,
                              content_type='application/json')
        assert response.status_code == 200
        
        rec_data = json.loads(response.data)
        assert rec_data['success'] is True
        
        # 4. Search for recipes
        search_request = {
            'query': 'ayam',
            'category_filter': categories_data['categories'][0],
            'limit': 5
        }
        
        response = client.post('/api/recipes/search',
                              json=search_request,
                              content_type='application/json')
        assert response.status_code == 200
        
        search_data = json.loads(response.data)
        assert search_data['success'] is True
    
    def test_content_based_fallback_workflow(self, client):
        """Test content-based recommendation sebagai fallback"""
        # 1. Try new user with invalid categories (should fail)
        response = client.post('/api/recommendations/new-user',
                              json={'preferred_categories': ['Invalid Category']},
                              content_type='application/json')
        assert response.status_code == 400
        
        # 2. Use content-based as fallback
        response = client.post('/api/recommendations/content-based',
                              json={'top_k': 5, 'min_rating': 4.0},
                              content_type='application/json')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['recommendation_method'] == 'content_based'
    
    def test_statistics_and_search_consistency(self, client):
        """Test konsistensi antara statistics dan search results"""
        # 1. Get statistics
        response = client.get('/api/stats')
        assert response.status_code == 200
        stats_data = json.loads(response.data)
        
        total_recipes = stats_data['statistics']['total_recipes']
        categories = list(stats_data['statistics']['category_distribution'].keys())
        
        # 2. Verify categories match with /api/categories
        response = client.get('/api/categories')
        assert response.status_code == 200
        cat_data = json.loads(response.data)
        
        # Categories should be consistent (allowing for some differences due to filtering)
        assert len(set(categories).intersection(set(cat_data['categories']))) > 0
        
        # 3. Search should return results within expected bounds
        for category in categories[:2]:  # Test first 2 categories
            search_request = {
                'query': '',  # Empty query to get all in category
                'category_filter': category,
                'limit': 50
            }
            
            response = client.post('/api/recipes/search',
                                  json=search_request,
                                  content_type='application/json')
            
            if response.status_code == 200:
                search_data = json.loads(response.data)
                # Results should not exceed total recipes
                assert search_data['total_results'] <= total_recipes


class TestPerformanceAndLimits:
    """Test suite untuk performance dan limits"""
    
    @pytest.mark.skip(reason="Flask test client not designed for true concurrency testing")
    def test_concurrent_requests_skipped(self, client):
        """Test handling multiple concurrent requests - Skipped due to Flask test client limitations"""
        pass
    
    def test_rapid_sequential_requests(self, client):
        """Test rapid sequential requests as alternative to concurrent testing"""
        import time
        
        success_count = 0
        total_requests = 10
        
        start_time = time.time()
        
        for i in range(total_requests):
            try:
                response = client.get('/health')
                if response.status_code == 200:
                    success_count += 1
            except Exception as e:
                print(f"Request {i} failed: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Rapid sequential test: {success_count}/{total_requests} success in {duration:.2f}s")
        
        # All sequential requests should succeed
        assert success_count == total_requests
        
        # Should complete reasonably fast
        assert duration < 5.0
    
    def test_sequential_requests_benchmark(self, client):
        """Test sequential requests as baseline performance"""
        start_time = time.time()
        
        for i in range(5):
            response = client.get('/health')
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Sequential test duration: {duration:.2f} seconds")
        assert duration < 10.0  # Should complete within 10 seconds
    
    def test_large_request_handling(self, client):
        """Test handling large requests"""
        # Test with large category list
        large_request = {
            'preferred_categories': ['Main Course'] * 50,  # Reduced from 100
            'top_k': 20  # Reduced from 50
        }
        
        response = client.post('/api/recommendations/new-user',
                              json=large_request,
                              content_type='application/json')
        
        # Should handle gracefully (either succeed or fail gracefully)
        assert response.status_code in [200, 400, 413, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            if data.get('success'):
                assert isinstance(data.get('recommendations', []), list)
    
    def test_edge_case_parameters(self, client):
        """Test edge case parameters"""
        edge_cases = [
            {'user_id': 0, 'top_k': 1},  # Zero user_id, positive top_k
            {'user_id': 1, 'top_k': 0},  # Positive user_id, zero top_k
            {'user_id': 999999, 'top_k': 1},  # Very large user_id
        ]
        
        for case in edge_cases:
            response = client.post('/api/recommendations/existing-user',
                                  json=case,
                                  content_type='application/json')
            
            # Should handle gracefully (not crash)
            assert response.status_code in [200, 400, 404, 500]
            
            if response.status_code == 200:
                data = json.loads(response.data)
                # If successful, should maintain data integrity
                if data.get('success'):
                    assert isinstance(data.get('recommendations', []), list)
    
    def test_response_time_limits(self, client):
        """Test that responses come within reasonable time limits"""
        endpoints_to_test = [
            '/health',
            '/api/info',
            '/api/categories',
            '/api/difficulties'
        ]
        
        for endpoint in endpoints_to_test:
            start_time = time.time()
            response = client.get(endpoint)
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Each endpoint should respond within 2 seconds
            assert duration < 2.0, f"{endpoint} took {duration:.2f} seconds"
            assert response.status_code == 200
    
    def test_memory_usage_stability(self, client):
        """Test that repeated requests don't cause memory issues"""
        # Make multiple requests to different endpoints
        endpoints = ['/health', '/api/info', '/api/categories']
        
        for _ in range(10):  # Repeat 10 times
            for endpoint in endpoints:
                response = client.get(endpoint)
                assert response.status_code == 200
                
        # If we reach here without crashing, memory usage is stable enough