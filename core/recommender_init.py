import os
import traceback
from scripts.recipe_recommender import EnhancedIndonesianRecipeRecommender
from core.logger import AppLogger


class RecommenderInitializer:
    """Handle recommender initialization and model loading"""
    
    def __init__(self):
        self.logger = AppLogger.get_logger(__name__)
        self.recommender = None
    
    def initialize_recommender(self):
        """Initialize and load the recommender model"""
        try:
            self.recommender = EnhancedIndonesianRecipeRecommender()
            self.logger.info("Recommender instance created successfully")
            
            if self._load_model():
                self.logger.info("Model loaded successfully!")
            else:
                self.logger.warning("Model files not found or failed to load. "
                                  "Recommender initialized but model not loaded.")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize recommender: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.recommender = None
        
        return self.recommender
    
    def _load_model(self):
        """Attempt to load model from various possible paths"""
        current_dir = os.getcwd()
        self.logger.info(f"Current working directory: {current_dir}")
        
        possible_paths = self._get_possible_model_paths(current_dir)
        
        for model_path in possible_paths:
            if self._try_load_from_path(model_path):
                return True
        
        self._log_debug_info(current_dir)
        return False
    
    def _get_possible_model_paths(self, current_dir):
        """Get list of possible model paths to try"""
        return [
            "models/recipe_recommender",
            "models/recipe_recommender_model.h5",
            os.path.join(current_dir, "models", "recipe_recommender"),
            os.path.join(current_dir, "models", "recipe_recommender_model.h5")
        ]
    
    def _try_load_from_path(self, model_path):
        """Try to load model from a specific path"""
        base_path = model_path.replace("_model.h5", "").replace(".h5", "")
        h5_path = f"{base_path}_model.h5" if not base_path.endswith("_model.h5") else base_path
        components_path = f"{base_path}_components.joblib"
        metadata_path = f"{base_path}_metadata.json"
        
        self.logger.info(f"Checking paths:")
        self.logger.info(f"  H5 model: {h5_path} - exists: {os.path.exists(h5_path)}")
        self.logger.info(f"  Components: {components_path} - exists: {os.path.exists(components_path)}")
        self.logger.info(f"  Metadata: {metadata_path} - exists: {os.path.exists(metadata_path)}")
        
        if os.path.exists(h5_path) and os.path.exists(components_path):
            self.logger.info(f"Found model files, attempting to load from: {base_path}")
            return self.recommender.load_model(base_path)
        
        return False
    
    def _log_debug_info(self, current_dir):
        """Log debug information about model directory"""
        models_dir = os.path.join(current_dir, "models")
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            self.logger.warning(f"Files in models directory: {files}")
        else:
            self.logger.warning(f"Models directory does not exist: {models_dir}")