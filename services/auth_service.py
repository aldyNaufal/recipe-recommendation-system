import jwt
import logging
from functools import wraps
from flask import request, jsonify, current_app
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AuthService:
    """Service untuk menangani autentikasi JWT"""
    
    @staticmethod
    def verify_token(token):
        """Verify JWT token"""
        try:
            print(token)
            print(current_app.config['JWT_SECRET_KEY']) 
            print(current_app.config['JWT_ALGORITHM']) 
            # Decode token menggunakan secret key yang sama dengan Node.js
            payload = jwt.decode(
                token, 
                
                current_app.config['JWT_SECRET_KEY'],
                 
                algorithms=[current_app.config['JWT_ALGORITHM']]
            )
            print(payload)
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return None
    
    @staticmethod
    def get_token_from_request():
        """Extract token from request headers"""
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return None
        
        # Format: "Bearer <token>"
        try:
            token = auth_header.split(' ')[1]
            return token
        except IndexError:
            logger.warning("Invalid Authorization header format")
            return None
    
    @staticmethod
    def get_current_user():
        """Get current user from token"""
        token = AuthService.get_token_from_request()
        
        if not token:
            return None
        
        payload = AuthService.verify_token(token)
        
        if not payload:
            return None
        
        return {
            'user_id': payload.get('user_id') or payload.get('userId'),
            'email': payload.get('email'),
            'username': payload.get('username'),
            'role': payload.get('role', 'user'),
            'exp': payload.get('exp'),
            'iat': payload.get('iat')
        }

def require_auth(f):
    """Decorator untuk memerlukan autentikasi"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            user = AuthService.get_current_user()
            
            if not user:
                return jsonify({
                    'success': False,
                    'error': 'Authentication required',
                    'code': 'AUTH_REQUIRED'
                }), 401
            
            # Add user to request context
            request.current_user = user
            
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Authentication failed',
                'code': 'AUTH_FAILED'
            }), 401
    
    return decorated_function

def optional_auth(f):
    """Decorator untuk autentikasi opsional"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            user = AuthService.get_current_user()
            request.current_user = user  # Bisa None
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Optional auth error: {str(e)}")
            request.current_user = None
            return f(*args, **kwargs)
    
    return decorated_function

def require_role(required_role):
    """Decorator untuk memerlukan role tertentu"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = AuthService.get_current_user()
            
            if not user:
                return jsonify({
                    'success': False,
                    'error': 'Authentication required',
                    'code': 'AUTH_REQUIRED'
                }), 401
            
            if user.get('role') != required_role:
                return jsonify({
                    'success': False,
                    'error': 'Insufficient permissions',
                    'code': 'INSUFFICIENT_PERMISSIONS'
                }), 403
            
            request.current_user = user
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator