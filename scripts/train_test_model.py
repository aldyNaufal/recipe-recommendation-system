
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from .recipe_recommender import EnhancedIndonesianRecipeRecommender

# Training dan Testing Functions
def train_enhanced_indonesian_recipe_model(df, test_size=0.2, validation_size=0.1, epochs=100, batch_size=512):
    """Fungsi utama untuk training enhanced model rekomendasi resep Indonesia"""
    
    print("=" * 60)
    print("🍽️  ENHANCED SISTEM REKOMENDASI RESEP MAKANAN INDONESIA")
    print("=" * 60)
    
    # Initialize enhanced recommender
    recommender = EnhancedIndonesianRecipeRecommender()
    
    # Preprocess data
    print("🔄 Memulai preprocessing data...")
    processed_data = recommender.preprocess_data(df)
    
    # Train model
    print("🚀 Memulai training model...")
    results = recommender.train_model(
        processed_data, 
        test_size=test_size,
        validation_size=validation_size,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Print training results
    print("\n📊 Training Results:")
    print(f"Train RMSE: {results['train_metrics']['rmse']:.4f}")
    print(f"Train MAE: {results['train_metrics']['mae']:.4f}")
    print(f"Val RMSE: {results['val_metrics']['rmse']:.4f}")
    print(f"Val MAE: {results['val_metrics']['mae']:.4f}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.4f}")
    print(f"Test MAE: {results['test_metrics']['mae']:.4f}")
    
    # Evaluate model
    print("\n📊 Evaluating model performance...")
    evaluation = recommender.evaluate_model(results['test_data'], results['test_indices'])
    
    # Validation criteria
    validation_criteria = {
        'rmse': 0.25,
        'mae': 0.20,
        'precision_at_10': 0.5,
        'hit_rate_at_10': 0.7,
        'diversity': 0.3
    }
    
    print("\n" + "=" * 60)
    print("📈 HASIL EVALUASI MODEL")
    print("=" * 60)
    print(f"RMSE: {evaluation['rmse']:.4f} (Target: < {validation_criteria['rmse']})")
    print(f"MAE: {evaluation['mae']:.4f} (Target: < {validation_criteria['mae']})")
    print(f"Precision@10: {evaluation['precision_at_10']:.4f} (Target: > {validation_criteria['precision_at_10']})")
    print(f"Hit Rate@10: {evaluation['hit_rate_at_10']:.4f} (Target: > {validation_criteria['hit_rate_at_10']})")
    print(f"Diversity: {evaluation['diversity']:.4f} (Target: > {validation_criteria['diversity']})")
    print(f"Users Evaluated: {evaluation['n_evaluated_users']}")
    
    # Check validation
    validation_passed = all([
        evaluation['rmse'] < validation_criteria['rmse'],
        evaluation['mae'] < validation_criteria['mae'],
        evaluation['precision_at_10'] > validation_criteria['precision_at_10'],
        evaluation['hit_rate_at_10'] > validation_criteria['hit_rate_at_10'],
        evaluation['diversity'] > validation_criteria['diversity']
    ])
    
    print("\n" + "=" * 60)
    print("✅ VALIDASI MODEL" if validation_passed else "❌ VALIDASI MODEL")
    print("=" * 60)
    
    if validation_passed:
        print("🎉 Model siap untuk production!")
        print("Model memenuhi semua kriteria kualitas untuk rekomendasi resep Indonesia.")
        recommender.save_model('../models/fix_model.pkl')
    else:
        print("⚠️  Model perlu perbaikan sebelum deployment.")
        print("💡 Saran perbaikan:")
        if evaluation['rmse'] >= validation_criteria['rmse']:
            print("   - Tingkatkan epochs atau sesuaikan learning rate")
        if evaluation['precision_at_10'] <= validation_criteria['precision_at_10']:
            print("   - Periksa kualitas data dan feature engineering")
        if evaluation['diversity'] <= validation_criteria['diversity']:
            print("   - Tambahkan diversity penalty dalam loss function")
    
    return {
        'recommender': recommender,
        'model': recommender.model,
        'processed_data': processed_data,
        'evaluation': evaluation,
        'validation_passed': validation_passed,
        'training_history': results['history'],
        'training_results': results
    }

def test_new_user_recommendations(recommender):
    """Test function untuk menguji rekomendasi user baru"""
    print("\n" + "=" * 60)
    print("🆕 TESTING REKOMENDASI UNTUK USER BARU")
    print("=" * 60)
    
    # Test case 1: User suka ayam dan ikan, mudah
    print("\n🧪 Test Case 1: User suka Ayam & Ikan, tingkat mudah")
    
    recommender = EnhancedIndonesianRecipeRecommender()

    recs1 = recommender.get_user_profile_based_recommendations(
        preferred_categories=['Ayam', 'Ikan'],
        preferred_difficulty='Cepat & Mudah',
        top_k=5,
        show_detailed=True
    )
    
    # Test case 2: User suka semua protein, tingkat sulit
    print("\n🧪 Test Case 2: User suka protein hewani, tingkat expert")
    recs2 = recommender.get_user_profile_based_recommendations(
        preferred_categories=['Sapi', 'Kambing', 'Udang'],
        preferred_difficulty='Level Dewa Masak',
        top_k=5,
        show_detailed=True
    )
    
    # Test case 3: User vegetarian
    print("\n🧪 Test Case 3: User vegetarian")
    recs3 = recommender.get_user_profile_based_recommendations(
        preferred_categories=['Tahu', 'Tempe', 'Telur'],
        preferred_difficulty='Butuh Usaha',
        top_k=5,
        show_detailed=True
    )
    
    return {
        'test_case_1': recs1,
        'test_case_2': recs2,
        'test_case_3': recs3
    }


def test_existing_user_recommendations(recommender, sample_users=5):
    """Test function untuk menguji rekomendasi existing users"""
    print("\n" + "=" * 60)
    print("👤 TESTING REKOMENDASI UNTUK EXISTING USERS")
    print("=" * 60)
    
    # Ambil sample users

    recommender = EnhancedIndonesianRecipeRecommender()

    all_users = recommender.processed_data['user_id'].unique()
    test_users = np.random.choice(all_users, min(sample_users, len(all_users)), replace=False)
    
    results = {}
    
    for user_id in test_users:
        print(f"\n🔍 Testing User ID: {user_id}")
        
        # Lihat history user
        user_history = recommender.processed_data[
            recommender.processed_data['user_id'] == user_id
        ][['Category_Mapped', 'rating', 'Difficulty_Level']].groupby(['Category_Mapped', 'Difficulty_Level']).agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        print("📊 User History Summary:")
        print(user_history)
        
        # Get recommendations
        recs = recommender.get_enhanced_recommendations(
            user_id, 
            top_k=5, 
            show_detailed=True
        )
        
        results[user_id] = {
            'history': user_history,
            'recommendations': recs
        }
    
    return results

# Example usage dan testing
def run_complete_training_and_testing(df):
    """
    Menjalankan complete training dan testing pipeline
    """
    print("🚀 Memulai Complete Training dan Testing Pipeline")
    print("=" * 80)
    
    # Step 1: Train model

    recommender = EnhancedIndonesianRecipeRecommender()

    training_results = train_enhanced_indonesian_recipe_model(
        df, 
        test_size=0.2, 
        validation_size=0.1, 
        epochs=50,  # Reduced for demo
        batch_size=512
    )
    
    recommender = training_results['recommender']
    
    # Step 2: Save model jika validasi passed
    if training_results['validation_passed']:
        print("\n💾 Menyimpan model...")
        recommender.save_model('enhanced_indonesian_recipe_model.pkl')
    
    # Step 3: Test new user recommendations
    print("\n" + "=" * 80)
    print("🧪 TESTING PHASE")
    print("=" * 80)
    
    new_user_tests = test_new_user_recommendations(recommender)
    existing_user_tests = test_existing_user_recommendations(recommender, sample_users=3)
    
    # Step 4: Summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"✅ Model Training: {'PASSED' if training_results['validation_passed'] else 'FAILED'}")
    print(f"📊 RMSE: {training_results['evaluation']['rmse']:.4f}")
    print(f"📊 Precision@10: {training_results['evaluation']['precision_at_10']:.4f}")
    print(f"👥 New User Tests: {len(new_user_tests)} test cases completed")
    print(f"👤 Existing User Tests: {len(existing_user_tests)} users tested")
    
    return {
        'training_results': training_results,
        'new_user_tests': new_user_tests,
        'existing_user_tests': existing_user_tests
    }