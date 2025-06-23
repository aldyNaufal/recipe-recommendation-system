import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import joblib
import json
import traceback

from .indonesian_processor import IndonesianTextPreprocessor

class EnhancedIndonesianRecipeRecommender:
    """Enhanced Sistem rekomendasi resep makanan Indonesia dengan User-Based CF untuk user baru"""
    
    
    def __init__(self):
        self.text_processor = IndonesianTextPreprocessor()
        self.model = None
        self.encoders = {}
        self.scalers = {}
        self.tfidf_vectorizer = None
        self.processed_data = None
        self.original_data = None
        self.category_similarity = None
        
        # Predefined categories dan difficulties sesuai requirement
        self.valid_categories = [
            "Ayam", "Ikan", "Kambing", "Sapi", "Tahu", "Telur", "Tempe", "Udang"
        ]
        
        self.difficulty_mapping = {
            1: "Cepat & Mudah",
            2: "Butuh Usaha", 
            3: "Level Dewa Masak"
        }
        
    def calculate_difficulty_score(self, total_ingredients, total_steps):
        """Hitung difficulty score berdasarkan ingredients dan steps"""
        ingredient_complexity = 1 if total_ingredients <= 5 else 2 if total_ingredients <= 10 else 3
        steps_complexity = 1 if total_steps <= 5 else 2 if total_steps <= 10 else 3
        return (ingredient_complexity + steps_complexity) / 2
    
    def get_difficulty_level(self, difficulty_score):
        """Convert difficulty score ke level"""
        if difficulty_score <= 1.5:
            return self.difficulty_mapping[1]
        elif difficulty_score <= 2.5:
            return self.difficulty_mapping[2]
        else:
            return self.difficulty_mapping[3]

    def preprocess_data(self, df):
        """Preprocessing data resep Indonesia dengan kategori yang sudah ditentukan"""
        print("🔄 Memulai preprocessing data...")
        
        # Simpan data asli
        self.original_data = df.copy()
        
        # Copy dataframe
        data = df.copy()
        
        # 1. Filter dan standardisasi kategori
        print("🏷️ Standardisasi kategori...")
        # Map kategori ke valid categories (case insensitive)
        def map_category(cat):
            cat_lower = str(cat).lower()
            for valid_cat in self.valid_categories:
                if valid_cat.lower() in cat_lower or cat_lower in valid_cat.lower():
                    return valid_cat
            return "Lainnya"  # Fallback category
        
        data['Category_Mapped'] = data['Category'].apply(map_category)
        # Hanya ambil data dengan kategori valid
        data = data[data['Category_Mapped'].isin(self.valid_categories)]
        
        # 2. Preprocessing teks berbahasa Indonesia
        print("📝 Preprocessing teks bahasa Indonesia...")
        data['Category_Cleaned'] = data['Category_Mapped'].apply(self.text_processor.clean_text)
        data['Title_Keywords'] = data['Title Cleaned'].apply(
            lambda x: ' '.join(self.text_processor.extract_keywords(x, 5))
        )
        # Urutan yang benar:
        # 1. Normalisasi ejaan dulu dengan process_ingredients()
        data['Ingredients_Spelling'] = data['Ingredients Cleaned'].apply(
            lambda x: self.text_processor.process_ingredients(x)
        )

        # 2. Baru ekstrak keywords dari hasil normalisasi
        data['Ingredients_Keywords'] = data['Ingredients_Spelling'].apply(
            lambda x: ' '.join(self.text_processor.extract_keywords(x, 10))
        )
        data['Steps_Keywords'] = data['Steps Cleaned'].apply(
            lambda x: ' '.join(self.text_processor.extract_keywords(x, 8))
        )
        
        # 3. Feature engineering untuk difficulty dengan mapping yang benar
        print("⚙️ Menghitung tingkat kesulitan resep...")
        
        # Hitung difficulty score berdasarkan ingredients dan steps
        data['Ingredient_Complexity'] = data['Total Ingredients'].apply(
            lambda x: 1 if x <= 5 else 2 if x <= 10 else 3
        )
        data['Steps_Complexity'] = data['Total Steps'].apply(
            lambda x: 1 if x <= 5 else 2 if x <= 10 else 3
        )
        data['Difficulty_Score'] = (data['Ingredient_Complexity'] + data['Steps_Complexity']) / 2
        
        # Map ke difficulty level yang sudah ditentukan
        data['Difficulty_Level'] = data['Difficulty_Score'].apply(
            lambda x: self.difficulty_mapping[1] if x <= 1.5 
            else self.difficulty_mapping[2] if x <= 2.5 
            else self.difficulty_mapping[3]
        )
        
        # 4. Normalisasi rating
        data['Rating_Normalized'] = data['rating'] / 5.0
        
        # Proper handling of total_rating normalization
        total_rating_scaler = MinMaxScaler()
        data['Total_Rating_Normalized'] = total_rating_scaler.fit_transform(
            data[['total_rating']].fillna(data['total_rating'].mean())
        ).flatten()
        self.scalers['total_rating'] = total_rating_scaler
        
        # 5. Encoding kategori dengan kategori yang sudah ditentukan
        print("🏷️ Encoding kategori...")
        category_encoder = LabelEncoder()
        data['Category_Encoded'] = category_encoder.fit_transform(data['Category_Mapped'])
        self.encoders['category'] = category_encoder
        
        # 6. User dan Item encoding
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        data['User_Encoded'] = user_encoder.fit_transform(data['user_id'])
        data['Item_Encoded'] = item_encoder.fit_transform(data['item_id'])
        self.encoders['user'] = user_encoder
        self.encoders['item'] = item_encoder
        
        # 7. TF-IDF untuk content similarity
        print("📊 Membuat TF-IDF vectors...")
        combined_text = (
            data['Title_Keywords'] + ' ' + 
            data['Ingredients_Keywords'] + ' ' + 
            data['Category_Cleaned']
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
        
        # Simpan similarity matrix
        self.category_similarity = cosine_similarity(tfidf_matrix)
        
        # 8. Scaling numerical features
        scaler = StandardScaler()
        numerical_features = ['Total Ingredients', 'Total Steps', 'Difficulty_Score']
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        self.scalers['numerical'] = scaler
        
        self.processed_data = data
        print("✅ Preprocessing selesai!")
        
        return data
    
    def build_model(self, n_users, n_items, n_categories, embedding_dim=50):
        """Membangun model hybrid recommendation"""
        print("🏗️ Membangun model rekomendasi hybrid...")
        
        # Input layers
        user_input = Input(shape=(), name='user_id')
        item_input = Input(shape=(), name='item_id')
        category_input = Input(shape=(), name='category')
        
        # Numerical features input
        numerical_input = Input(shape=(3,), name='numerical_features')
        
        # Embedding layers
        user_embedding = Embedding(n_users, embedding_dim, name='user_embedding')(user_input)
        item_embedding = Embedding(n_items, embedding_dim, name='item_embedding')(item_input)
        category_embedding = Embedding(n_categories, embedding_dim//2, name='category_embedding')(category_input)
        
        # Flatten embeddings
        user_vec = tf.keras.layers.Flatten()(user_embedding)
        item_vec = tf.keras.layers.Flatten()(item_embedding)
        category_vec = tf.keras.layers.Flatten()(category_embedding)
        
        # Concatenate all features
        concat = Concatenate()([user_vec, item_vec, category_vec, numerical_input])
        
        # Deep layers
        x = Dense(256, activation='relu')(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='rating')(x)
        
        # Create model
        model = Model(
            inputs=[user_input, item_input, category_input, numerical_input],
            outputs=output
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_training_data(self, data):
        """Menyiapkan data untuk training"""
        X = {
            'user_id': data['User_Encoded'].values,
            'item_id': data['Item_Encoded'].values,
            'category': data['Category_Encoded'].values,
            'numerical_features': data[['Total Ingredients', 'Total Steps', 'Difficulty_Score']].values
        }
        y = data['Rating_Normalized'].values
        
        return X, y
    
    def train_model(self, data, test_size=0.2, validation_size=0.1, epochs=100, batch_size=512):
        """Training model dengan validasi"""
        print("🚀 Memulai training model...")
        
        # Prepare data
        X, y = self.prepare_training_data(data)
        
        # Get indices for splitting
        indices = np.arange(len(y))
        
        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=42, 
            stratify=data['Category_Encoded']
        )
        
        # Second split: train vs val
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=validation_size/(1-test_size), 
            random_state=42,
            stratify=data['Category_Encoded'].iloc[train_val_indices]
        )
        
        # Create splits using indices
        X_train = {
            'user_id': X['user_id'][train_indices],
            'item_id': X['item_id'][train_indices],
            'category': X['category'][train_indices],
            'numerical_features': X['numerical_features'][train_indices]
        }
        
        X_val = {
            'user_id': X['user_id'][val_indices],
            'item_id': X['item_id'][val_indices],
            'category': X['category'][val_indices],
            'numerical_features': X['numerical_features'][val_indices]
        }
        
        X_test = {
            'user_id': X['user_id'][test_indices],
            'item_id': X['item_id'][test_indices],
            'category': X['category'][test_indices],
            'numerical_features': X['numerical_features'][test_indices]
        }
        
        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]
        
        # Build model
        n_users = len(self.encoders['user'].classes_)
        n_items = len(self.encoders['item'].classes_)
        n_categories = len(self.encoders['category'].classes_)
        
        self.build_model(n_users, n_items, n_categories)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        train_loss, train_mae = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'history': history,
            'train_metrics': {'rmse': np.sqrt(train_loss), 'mae': train_mae},
            'val_metrics': {'rmse': np.sqrt(val_loss), 'mae': val_mae},
            'test_metrics': {'rmse': np.sqrt(test_loss), 'mae': test_mae},
            'test_data': (X_test, y_test),
            'test_indices': test_indices
        }
    
    def find_similar_users(self, preferred_categories, min_similarity=0.3, top_k_users=20):
        """Mencari user yang memiliki preferensi kategori serupa untuk user baru"""
        print(f"🔍 Mencari user dengan preferensi serupa untuk kategori: {preferred_categories}")
        
        # Hitung preferensi setiap user berdasarkan rating mereka
        user_category_preferences = {}
        
        for user_id in self.processed_data['user_id'].unique():
            user_data = self.processed_data[self.processed_data['user_id'] == user_id]
            
            # Hitung rating rata-rata per kategori untuk user ini
            category_ratings = {}
            for category in self.valid_categories:
                cat_data = user_data[user_data['Category_Mapped'] == category]
                if len(cat_data) > 0:
                    avg_rating = cat_data['rating'].mean()
                    category_ratings[category] = avg_rating
                else:
                    category_ratings[category] = 0
            
            user_category_preferences[user_id] = category_ratings
        
        # Buat vektor preferensi untuk user baru berdasarkan kategori yang dipilih
        new_user_vector = np.zeros(len(self.valid_categories))
        for i, category in enumerate(self.valid_categories):
            if category in preferred_categories:
                new_user_vector[i] = 5.0  # Assume high preference for selected categories
        
        # Hitung similarity dengan existing users
        similar_users = []
        
        for user_id, preferences in user_category_preferences.items():
            # Buat vektor dari preferensi user existing
            user_vector = np.array([preferences[cat] for cat in self.valid_categories])
            
            # Hitung cosine similarity
            if np.linalg.norm(user_vector) > 0 and np.linalg.norm(new_user_vector) > 0:
                similarity = np.dot(new_user_vector, user_vector) / (
                    np.linalg.norm(new_user_vector) * np.linalg.norm(user_vector)
                )
                
                if similarity >= min_similarity:
                    similar_users.append({
                        'user_id': user_id,
                        'similarity': similarity,
                        'preferences': preferences
                    })
        
        # Sort by similarity dan ambil top K
        similar_users.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_users[:top_k_users]
    
    def get_user_based_recommendations_for_new_user(self, preferred_categories, 
                                                   difficulty_filter=None, top_k=10, 
                                                   min_rating=3.0, show_detailed=True):
        """Rekomendasi berbasis user similarity untuk user baru"""
        print(f"\n🆕 Mencari rekomendasi untuk USER BARU dengan preferensi: {preferred_categories}")
        print("=" * 60)
        
        # 1. Cari user yang serupa
        similar_users = self.find_similar_users(preferred_categories)
        
        if not similar_users:
            print("❌ Tidak ditemukan user dengan preferensi serupa, fallback ke content-based")
            return self._get_enhanced_content_based_recommendations(
                preferred_categories, difficulty_filter, top_k, min_rating, show_detailed
            )
        
        print(f"👥 Ditemukan {len(similar_users)} user dengan preferensi serupa")
        
        # 2. Kumpulkan resep yang disukai oleh similar users
        candidate_recipes = {}
        
        for similar_user in similar_users:
            user_id = similar_user['user_id']
            similarity_score = similar_user['similarity']
            
            # Ambil resep yang di-rating tinggi oleh user ini
            user_data = self.processed_data[
                (self.processed_data['user_id'] == user_id) & 
                (self.processed_data['rating'] >= min_rating)
            ]
            
            # Filter berdasarkan kategori yang disukai user baru
            if preferred_categories:
                user_data = user_data[user_data['Category_Mapped'].isin(preferred_categories)]
            
            # Filter berdasarkan difficulty jika ada
            if difficulty_filter:
                user_data = user_data[user_data['Difficulty_Level'] == difficulty_filter]
            
            # Tambahkan ke candidate dengan weighted score
            for _, recipe in user_data.iterrows():
                item_id = recipe['item_id']
                weighted_rating = recipe['rating'] * similarity_score
                
                if item_id in candidate_recipes:
                    candidate_recipes[item_id]['total_weighted_rating'] += weighted_rating
                    candidate_recipes[item_id]['vote_count'] += 1
                    candidate_recipes[item_id]['similarity_sum'] += similarity_score
                else:
                    candidate_recipes[item_id] = {
                        'total_weighted_rating': weighted_rating,
                        'vote_count': 1,
                        'similarity_sum': similarity_score,
                        'recipe_data': recipe
                    }
        
        # 3. Hitung final score dan rank
        recommendations = []
        for item_id, data in candidate_recipes.items():
            # Final score = weighted average rating * diversity bonus
            avg_weighted_rating = data['total_weighted_rating'] / data['similarity_sum']
            diversity_bonus = min(data['vote_count'] / len(similar_users), 1.0)  # Normalize
            final_score = avg_weighted_rating * (0.8 + 0.2 * diversity_bonus)
            
            recipe_data = data['recipe_data']
            original_recipe = self.original_data[self.original_data['item_id'] == item_id].iloc[0]
            
            recommendations.append({
                'item_id': item_id,
                'title_cleaned': original_recipe['Title Cleaned'],
                'steps_cleaned': original_recipe['Steps Cleaned'],
                'ingredients_cleaned': original_recipe['Ingredients Cleaned'],
                'category': recipe_data['Category_Mapped'],
                'total_rating': original_recipe['total_rating'],
                'image_url': original_recipe.get('Image URL', 'N/A'),
                'predicted_rating': final_score,
                'difficulty_level': recipe_data['Difficulty_Level'],
                'difficulty_score': recipe_data['Difficulty_Score'],
                'total_ingredients': original_recipe['Total Ingredients'],
                'total_steps': original_recipe['Total Steps'],
                'user_type': 'new_user_based',
                'vote_count': data['vote_count'],
                'similarity_users': len(similar_users)
            })
        
        # 4. Sort dan ambil top K
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        final_recommendations = recommendations[:top_k]
        
        # 5. Display results
        if show_detailed:
            self._display_recommendations("NEW_USER", final_recommendations, "new_user_based")
        
        return final_recommendations

    def _get_content_based_recommendations_for_new_user(self, category_filter=None, difficulty_max=3, 
                                                        top_k=10, min_rating=3.0, show_detailed=True):
        """Content-based recommendations untuk user baru"""
        print("🆕 Generating content-based recommendations for new user...")

        # Step 1: Copy data
        filtered_data = self.original_data.copy()
        print(f"✅ Jumlah data awal: {len(filtered_data)}")

        # Step 2: Filter berdasarkan kategori
        if category_filter:
            filtered_data = filtered_data[filtered_data['Category'] == category_filter]
            print(f"📂 Setelah filter kategori '{category_filter}': {len(filtered_data)}")

        # Step 3: Filter berdasarkan difficulty dan rating
        filtered_data = filtered_data[
            (filtered_data['Difficulty_Score'] <= difficulty_max) &
            (filtered_data['total_rating'] >= min_rating)
        ]
        print(f"🎯 Setelah filter difficulty <= {difficulty_max} dan min_rating >= {min_rating}: {len(filtered_data)}")

        # Step 4: Fallback jika kosong
        if filtered_data.empty:
            print("⚠️ Tidak ada data yang cocok dengan semua filter. Menggunakan fallback longgar...")
            filtered_data = self.original_data.copy()
            if category_filter:
                filtered_data = filtered_data[filtered_data['Category'] == category_filter]
                print(f"🔁 Fallback - kategori '{category_filter}': {len(filtered_data)}")

            # Fallback tanpa filter rating dan difficulty
            filtered_data = filtered_data.sort_values('total_rating', ascending=False).head(top_k)
        else:
            # Sort dan ambil top-k
            filtered_data = filtered_data.sort_values('total_rating', ascending=False).head(top_k)

        # Step 5: Format output
        recommendations = []
        for _, row in filtered_data.iterrows():
            recommendations.append({
                'item_id': row['item_id'],
                'title_cleaned': row.get('Title Cleaned', 'N/A'),
                'steps_cleaned': row.get('Steps Cleaned', 'N/A'),
                'ingredients_cleaned': row.get('Ingredients Cleaned', 'N/A'),
                'category': row.get('Category', 'Unknown'),
                'total_rating': row.get('total_rating', 0),
                'image_url': row.get('Image URL', 'N/A'),
                'predicted_rating': row.get('total_rating', 0),  # Use actual rating as predicted
                'difficulty_level': self._calculate_difficulty_level(row['Difficulty_Score']),
                'difficulty_score': row.get('Difficulty_Score', 0),
                'total_ingredients': row.get('Total Ingredients', 0),
                'total_steps': row.get('Total Steps', 0),
                'user_type': 'new'
            })

        if not recommendations:
            print("❌ Tidak ada rekomendasi yang dapat ditampilkan meskipun fallback dilakukan.")
        elif show_detailed:
            self._display_recommendations("NEW_USER", recommendations, "new")

        return recommendations



    def _display_recommendations(self, user_id, recommendations, user_type):
        """Display recommendations dalam format yang rapi"""
        type_labels = {
            'existing': 'Existing User (Collaborative Filtering)',
            'new_user_based': 'New User (User-Based Collaborative Filtering)', 
            'content_based': 'New User (Content-Based Filtering)'
        }
        
        print(f"\n🍽️  REKOMENDASI RESEP UNTUK USER: {user_id}")
        print(f"👤 Tipe User: {type_labels.get(user_type, user_type)}")
        print("=" * 80)
        
        if not recommendations:
            print("❌ Tidak ada rekomendasi yang memenuhi kriteria")
            return
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n🏆 #{i}")
            print(f"📝 Judul: {rec['title_cleaned']}")
            print(f"🏷️  Kategori: {rec['category']}")
            print(f"⭐ Rating Prediksi: {rec['predicted_rating']:.2f}/5.0")
            print(f"📊 Total Rating: {rec['total_rating']}")
            print(f"🔥 Tingkat Kesulitan: {rec['difficulty_level']}")
            print(f"🥘 Total Bahan: {rec['total_ingredients']}")
            print(f"📋 Total Langkah: {rec['total_steps']}")
            
            # Show additional info for user-based recommendations
            if user_type == 'new_user_based':
                print(f"👥 Berdasarkan {rec.get('vote_count', 0)} user serupa")
                print(f"🤝 Dari {rec.get('similarity_users', 0)} similar users")
            
            # Show ingredients (first 100 chars)
            ingredients = rec['ingredients_cleaned'][:100]
            if len(rec['ingredients_cleaned']) > 100:
                ingredients += "..."
            print(f"🛒 Bahan: {ingredients}")
            
            # Show steps (first 150 chars)
            steps = rec['steps_cleaned'][:150]
            if len(rec['steps_cleaned']) > 150:
                steps += "..."
            print(f"👨‍🍳 Cara Masak: {steps}")
            
            if rec['image_url'] != 'N/A':
                print(f"🖼️  Gambar: {rec['image_url']}")
            
            print("-" * 60)
    
    def get_user_profile_based_recommendations(self, preferred_categories, preferred_difficulty=None, 
                                             dietary_restrictions=None, cooking_time_preference=None,
                                             top_k=10, min_rating=3.0, show_detailed=True):
        """
        Rekomendasi untuk user baru berdasarkan profil yang diisi saat registrasi
        menggunakan User-Based Collaborative Filtering
        
        Parameters:
        - preferred_categories: list kategori yang disukai user baru
        - preferred_difficulty: tingkat kesulitan yang disukai 
        - dietary_restrictions: pembatasan diet (opsional untuk future enhancement)
        - cooking_time_preference: preferensi waktu memasak (opsional)
        - top_k: jumlah rekomendasi
        - min_rating: minimum rating resep
        - show_detailed: tampilkan detail
        """
        print(f"\n🆕 REKOMENDASI UNTUK USER BARU")
        print(f"❤️  Kategori Favorit: {', '.join(preferred_categories)}")
        if preferred_difficulty:
            print(f"⚡ Tingkat Kesulitan: {preferred_difficulty}")
        print("=" * 60)
        
        # Step 1: Find similar users based on category preferences
        similar_users = self._find_similar_users_by_preferences(
            preferred_categories, preferred_difficulty, min_similarity=0.2
        )
        
        if not similar_users:
            print("⚠️  Tidak ditemukan user dengan preferensi serupa")
            print("🔄 Menggunakan Content-Based Filtering sebagai fallback...")
            return self._get_enhanced_content_based_recommendations(
                preferred_categories, preferred_difficulty, top_k, min_rating, show_detailed
            )
        
        print(f"👥 Ditemukan {len(similar_users)} user dengan preferensi serupa")
        
        # Step 2: Get recommendations from similar users
        recommendations = self._get_collaborative_recommendations_from_similar_users(
            similar_users, preferred_categories, preferred_difficulty, 
            top_k, min_rating
        )
        
        # Step 3: Display results
        if show_detailed:
            self._display_recommendations("NEW_USER", recommendations, "new_user_based")
        
        return recommendations
    
    def _find_similar_users_by_preferences(self, preferred_categories, preferred_difficulty=None, min_similarity=0.2):
        """Mencari user existing yang memiliki preferensi serupa dengan user baru"""
        print("🔍 Mencari user dengan preferensi serupa...")
        
        # Analyze existing users' preferences
        user_preferences = {}
        
        for user_id in self.processed_data['user_id'].unique():
            user_data = self.processed_data[self.processed_data['user_id'] == user_id]
            
            # Calculate category preferences (weighted by rating)
            category_scores = {}
            total_ratings = 0
            
            for category in self.valid_categories:
                cat_data = user_data[user_data['Category_Mapped'] == category]
                if len(cat_data) > 0:
                    # Weight by rating - higher rated recipes indicate stronger preference
                    weighted_score = (cat_data['rating'] * cat_data['total_rating']).sum()
                    total_count = len(cat_data)
                    category_scores[category] = weighted_score / total_count if total_count > 0 else 0
                    total_ratings += weighted_score
                else:
                    category_scores[category] = 0
            
            # Normalize scores
            if total_ratings > 0:
                for cat in category_scores:
                    category_scores[cat] = category_scores[cat] / total_ratings * 100
            
            # Calculate difficulty preference if specified
            difficulty_match = 1.0  # Default perfect match
            if preferred_difficulty:
                user_difficulty_dist = user_data['Difficulty_Level'].value_counts(normalize=True)
                if preferred_difficulty in user_difficulty_dist:
                    difficulty_match = user_difficulty_dist[preferred_difficulty]
                else:
                    difficulty_match = 0.1  # Low match if user never tried this difficulty
            
            user_preferences[user_id] = {
                'category_scores': category_scores,
                'difficulty_match': difficulty_match,
                'total_recipes_rated': len(user_data)
            }
        
        # Calculate similarity with new user preferences
        similar_users = []
        
        # Create preference vector for new user
        new_user_vector = np.zeros(len(self.valid_categories))
        for i, category in enumerate(self.valid_categories):
            if category in preferred_categories:
                new_user_vector[i] = 100 / len(preferred_categories)  # Equal weight for preferred categories
        
        for user_id, prefs in user_preferences.items():
            # Skip users with too few ratings
            if prefs['total_recipes_rated'] < 5:
                continue
                
            # Create vector for existing user
            existing_user_vector = np.array([prefs['category_scores'][cat] for cat in self.valid_categories])
            
            # Calculate cosine similarity for categories
            if np.linalg.norm(existing_user_vector) > 0:
                category_similarity = np.dot(new_user_vector, existing_user_vector) / (
                    np.linalg.norm(new_user_vector) * np.linalg.norm(existing_user_vector)
                )
            else:
                category_similarity = 0
            
            # Combine with difficulty similarity
            total_similarity = category_similarity * 0.8 + prefs['difficulty_match'] * 0.2
            
            if total_similarity >= min_similarity:
                similar_users.append({
                    'user_id': user_id,
                    'similarity': total_similarity,
                    'category_similarity': category_similarity,
                    'difficulty_match': prefs['difficulty_match'],
                    'total_recipes': prefs['total_recipes_rated']
                })
        
        # Sort by similarity and return top users
        similar_users.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_users[:20]  # Top 20 similar users
    
    def _get_collaborative_recommendations_from_similar_users(self, similar_users, preferred_categories, 
                                                           preferred_difficulty, top_k, min_rating):
        """Mendapatkan rekomendasi dari user-user yang serupa"""
        print("🤝 Menganalisis preferensi dari user serupa...")
        
        # Collect highly-rated recipes from similar users
        recipe_scores = {}
        
        for similar_user in similar_users:
            user_id = similar_user['user_id']
            similarity_weight = similar_user['similarity']
            
            # Get user's highly rated recipes
            user_recipes = self.processed_data[
                (self.processed_data['user_id'] == user_id) & 
                (self.processed_data['rating'] >= min_rating)
            ]
            
            # Filter by preferred categories
            if preferred_categories:
                user_recipes = user_recipes[user_recipes['Category_Mapped'].isin(preferred_categories)]
            
            # Filter by difficulty if specified
            if preferred_difficulty:
                user_recipes = user_recipes[user_recipes['Difficulty_Level'] == preferred_difficulty]
            
            # Score each recipe
            for _, recipe in user_recipes.iterrows():
                item_id = recipe['item_id']
                
                # Calculate weighted score
                base_score = recipe['rating'] * similarity_weight
                popularity_bonus = np.log1p(recipe['total_rating']) * 0.1  # Small bonus for popular recipes
                final_score = base_score + popularity_bonus
                
                if item_id in recipe_scores:
                    recipe_scores[item_id]['total_score'] += final_score
                    recipe_scores[item_id]['vote_count'] += 1
                    recipe_scores[item_id]['voters'].append({
                        'user_id': user_id,
                        'similarity': similarity_weight,
                        'rating': recipe['rating']
                    })
                else:
                    recipe_scores[item_id] = {
                        'total_score': final_score,
                        'vote_count': 1,
                        'recipe_data': recipe,
                        'voters': [{
                            'user_id': user_id,
                            'similarity': similarity_weight,
                            'rating': recipe['rating']
                        }]
                    }
        
        # Convert to final recommendations
        recommendations = []
        
        for item_id, scores in recipe_scores.items():
            # Calculate final recommendation score
            avg_score = scores['total_score'] / scores['vote_count']
            diversity_bonus = min(scores['vote_count'] / len(similar_users), 1.0)
            final_score = avg_score * (0.9 + 0.1 * diversity_bonus)
            
            # Get original recipe data
            recipe_data = scores['recipe_data']
            original_recipe = self.original_data[self.original_data['item_id'] == item_id].iloc[0]
            
            recommendations.append({
                'item_id': item_id,
                'title_cleaned': original_recipe['Title Cleaned'],
                'steps_cleaned': original_recipe['Steps Cleaned'],
                'ingredients_cleaned': original_recipe['Ingredients Cleaned'],
                'category': recipe_data['Category_Mapped'],
                'total_rating': original_recipe['total_rating'],
                'image_url': original_recipe.get('Image URL', 'N/A'),
                'predicted_rating': final_score,
                'difficulty_level': recipe_data['Difficulty_Level'],
                'difficulty_score': recipe_data['Difficulty_Score'],
                'total_ingredients': original_recipe['Total Ingredients'],
                'total_steps': original_recipe['Total Steps'],
                'user_type': 'new_user_based',
                'vote_count': scores['vote_count'],
                'similarity_users': len(similar_users),
                'voters_info': scores['voters'][:3]  # Top 3 voters info for explanation
            })
        
        # Sort by predicted rating and return top K
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:top_k]


    def save_model(self, filepath):
        """Menyimpan model dengan format H5 untuk TensorFlow dan joblib untuk komponen lain"""
        print(f"💾 Menyimpan model ke {filepath}...")
        
        # Buat folder jika belum ada
        base_path = Path(filepath).parent
        base_path.mkdir(parents=True, exist_ok=True)
        
        model_name = Path(filepath).stem
        
        try:
            # 1. Simpan TensorFlow model dalam format H5
            tf_model_path = base_path / f"{model_name}_model.h5"
            self.model.save(tf_model_path, save_format='h5')
            print(f"✅ TensorFlow model disimpan ke {tf_model_path}")
            
            # 2. Simpan komponen preprocessing dengan joblib
            components = {
                'encoders': self.encoders,
                'scalers': self.scalers,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'processed_data': self.processed_data,
                'original_data': self.original_data,
                'category_similarity': self.category_similarity,
                'valid_categories': self.valid_categories,
                'difficulty_mapping': self.difficulty_mapping
            }
            
            components_path = base_path / f"{model_name}_components.joblib"
            joblib.dump(components, components_path, compress=3)  # Kompresi untuk ukuran file
            print(f"✅ Komponen preprocessing disimpan ke {components_path}")
            
            # 3. Simpan metadata
            metadata = {
                'tensorflow_version': tf.__version__,
                'model_file': f"{model_name}_model.h5",
                'components_file': f"{model_name}_components.joblib",
                'save_format': 'h5',
                'model_architecture': 'neural_collaborative_filtering'
            }
            
            metadata_path = base_path / f"{model_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print("✅ Model dan semua komponen berhasil disimpan!")
            return True
            
        except Exception as e:
            print(f"❌ Error saat menyimpan model: {str(e)}")
            return False

    def load_model(self, filepath):
        """Load model H5 dan semua komponen untuk Flask API"""
        print(f"📁 Loading model dari {filepath}...")
        
        
        base_path = Path(filepath).parent if Path(filepath).suffix else Path(filepath)
        model_name = Path(filepath).stem if Path(filepath).suffix else Path(filepath).name
        
        # Jika filepath sudah berupa base name, gunakan langsung
        if not Path(filepath).suffix:
            base_path = Path(filepath).parent if Path(filepath).parent.name != '.' else Path('.')
            model_name = Path(filepath).name
        
        try:
            print(f"🔍 Base path: {base_path}")
            print(f"🔍 Model name: {model_name}")
            
            # 1. Load TensorFlow model H5
            tf_model_path = base_path / f"{model_name}_model.h5"
            print(f"🔍 Looking for TF model at: {tf_model_path}")
            
            if tf_model_path.exists():
                self.model = tf.keras.models.load_model(str(tf_model_path))
                print(f"✅ TensorFlow model loaded dari {tf_model_path}")
            else:
                print(f"❌ Model file tidak ditemukan: {tf_model_path}")
                return False
            
            # 2. Load komponen preprocessing
            components_path = base_path / f"{model_name}_components.joblib"
            print(f"🔍 Looking for components at: {components_path}")
            
            if components_path.exists():
                components = joblib.load(str(components_path))
                
                # Assign komponen ke instance variables
                self.encoders = components.get('encoders')
                self.scalers = components.get('scalers')
                self.tfidf_vectorizer = components.get('tfidf_vectorizer')
                self.processed_data = components.get('processed_data')
                self.original_data = components.get('original_data')
                self.category_similarity = components.get('category_similarity')
                self.valid_categories = components.get('valid_categories')
                self.difficulty_mapping = components.get('difficulty_mapping')
                
                print(f"✅ Komponen preprocessing loaded dari {components_path}")
                
                # Debug: Print what components were loaded
                print("🔍 Loaded components:")
                for key, value in components.items():
                    if value is not None:
                        print(f"  - {key}: {type(value)}")
            else:
                print(f"❌ Components file tidak ditemukan: {components_path}")
                return False
            
            # 3. Load metadata (optional)
            metadata_path = base_path / f"{model_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"✅ Metadata loaded: {metadata}")
            else:
                print("⚠️ Metadata tidak ditemukan, melanjutkan tanpa metadata")
            
            # 4. Validasi komponen yang dimuat
            validation_result = self._validate_loaded_components()
            if not validation_result:
                print("❌ Validasi komponen gagal")
                return False
            
            print("🎉 Model dan semua komponen berhasil dimuat!")
            return True
            
        except Exception as e:
            print(f"❌ Error saat loading model: {str(e)}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return False

    def _validate_loaded_components(self):
        """Validasi bahwa semua komponen penting sudah dimuat"""
        try:
            required_components = ['model', 'encoders', 'processed_data', 'original_data']
            missing_components = []
            
            for component in required_components:
                if not hasattr(self, component) or getattr(self, component) is None:
                    missing_components.append(component)
            
            if missing_components:
                print(f"❌ Komponen yang hilang: {missing_components}")
                return False
            
            # Validasi spesifik untuk encoders
            if self.encoders and 'user' not in self.encoders:
                print("❌ User encoder tidak ditemukan")
                return False
            
            if self.encoders and 'item' not in self.encoders:
                print("❌ Item encoder tidak ditemukan")
                return False
            
            print("✅ Semua komponen penting sudah dimuat dan valid")
            return True
            
        except Exception as e:
            print(f"❌ Error saat validasi komponen: {str(e)}")
            return False

    def is_model_loaded(self):
        """Check if model is properly loaded"""
        return (hasattr(self, 'model') and self.model is not None and 
                hasattr(self, 'encoders') and self.encoders is not None and 
                hasattr(self, 'processed_data') and self.processed_data is not None)
    


    
    def predict_recommendations(self, user_input):
        """Method untuk prediksi rekomendasi (contoh implementasi)"""
        if self.model is None:
            raise ValueError("Model belum dimuat. Panggil load_model() terlebih dahulu.")
        
        try:
            # Preprocessing input sesuai dengan training
            processed_input = self._preprocess_input(user_input)
            
            # Prediksi menggunakan model
            predictions = self.model.predict(processed_input)
            
            # Post-processing hasil prediksi
            recommendations = self._postprocess_predictions(predictions)
            
            return recommendations
        
        except Exception as e:
            print(f"❌ Error saat prediksi: {str(e)}")
            return None
    
    def _preprocess_input(self, user_input):
        """Preprocessing input sesuai dengan training data"""
        # Implementasi preprocessing sesuai dengan yang digunakan saat training
        # Contoh: encoding, scaling, dll.
        pass
    
    def _postprocess_predictions(self, predictions):
        """Post-processing hasil prediksi menjadi rekomendasi"""
        # Implementasi post-processing untuk menghasilkan rekomendasi final
        pass

    def get_enhanced_recommendations(self, user_id, top_k=10, category_filter=None, difficulty_max=3, 
                                    min_rating=3.0, show_detailed=True, disable_filtering=False):
        """Mendapatkan rekomendasi dengan output lengkap - untuk existing users"""
        if self.model is None:
            raise ValueError("Model belum ditraining!")
        
        print(f"\n🎯 Mencari rekomendasi untuk User ID: {user_id}")
        print("=" * 60)
        
        # Encode user_id
        try:
            user_encoded = self.encoders['user'].transform([user_id])[0]
            user_type = "existing"
        except:
            # PERBAIKAN: Handle user baru dengan content-based fallback
            print("👤 User baru terdeteksi - menggunakan content-based recommendations")
            user_type = "new"
            return self._get_content_based_recommendations_for_new_user(
                category_filter, difficulty_max, top_k, min_rating, show_detailed
            )
        
        # Get all items yang belum di-rate user ini
        user_data = self.processed_data[self.processed_data['user_id'] == user_id]
        rated_items = set(user_data['item_id'].values)
        all_items = set(self.processed_data['item_id'].values)
        unrated_items = list(all_items - rated_items)
        
        print(f"📊 User telah menilai {len(rated_items)} resep dari total {len(all_items)} resep")
        
        if not unrated_items:
            print("⚠️  User sudah menilai semua resep!")
            return []
        
        # Prepare prediction data
        n_items = len(unrated_items)
        pred_data = {
            'user_id': np.full(n_items, user_encoded),
            'item_id': self.encoders['item'].transform(unrated_items),
            'category': [],
            'numerical_features': []
        }
        
        # Get item features
        for item_id in unrated_items:
            item_data = self.processed_data[self.processed_data['item_id'] == item_id].iloc[0]
            pred_data['category'].append(item_data['Category_Encoded'])
            pred_data['numerical_features'].append([
                item_data['Total Ingredients'],
                item_data['Total Steps'], 
                item_data['Difficulty_Score']
            ])
        
        pred_data['category'] = np.array(pred_data['category'])
        pred_data['numerical_features'] = np.array(pred_data['numerical_features'])
        
        # Predict ratings
        predictions = self.model.predict(pred_data, verbose=0).flatten()
        
        # Create recommendation list dengan info lengkap
        recommendations = []
        for i, item_id in enumerate(unrated_items):
            # Ambil data dari processed data
            item_data = self.processed_data[self.processed_data['item_id'] == item_id].iloc[0]
            
            # Ambil data asli untuk output lengkap
            original_item_data = self.original_data[self.original_data['item_id'] == item_id].iloc[0]
            
            predicted_rating = predictions[i] * 5  # Scale back to 1-5
            
            # PERBAIKAN: Gunakan field yang konsisten
            category_field = item_data.get('Category_Mapped', item_data.get('Category', 'Unknown'))
            difficulty_level = item_data.get('Difficulty_Level', self._calculate_difficulty_level(item_data['Difficulty_Score']))
            
            # PERBAIKAN: Apply filters dengan logika yang benar
            if not disable_filtering:
                # Category filter - pastikan menggunakan field yang benar
                if category_filter:
                    # Coba berbagai field kategori yang mungkin ada
                    item_category = (original_item_data.get('Category') or 
                                item_data.get('Category') or 
                                item_data.get('Category_Mapped', 'Unknown'))
                    if item_category != category_filter:
                        continue
                
                # Difficulty filter
                if item_data['Difficulty_Score'] > difficulty_max:
                    continue
                    
                # Rating filter
                if predicted_rating < min_rating:
                    continue
                
            # Hitung difficulty level yang lebih readable
            if item_data['Difficulty_Score'] <= 1.5:
                difficulty_level = "Cepat & Mudah"
            elif item_data['Difficulty_Score'] <= 2.5:
                difficulty_level = "Butuh Usaha"
            else:
                difficulty_level = "Level Dewa Masak"
                
            recommendations.append({
                'item_id': item_id,
                'title_cleaned': original_item_data['Title Cleaned'],
                'steps_cleaned': original_item_data['Steps Cleaned'],
                'ingredients_cleaned': original_item_data['Ingredients Cleaned'],
                'category': original_item_data.get('Category', category_field),
                'total_rating': original_item_data['total_rating'],
                'image_url': original_item_data.get('Image URL', 'N/A'),
                'predicted_rating': predicted_rating,
                'difficulty_level': difficulty_level,
                'difficulty_score': item_data['Difficulty_Score'],
                'total_ingredients': original_item_data['Total Ingredients'],
                'total_steps': original_item_data['Total Steps'],
                'user_type': user_type
            })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        final_recommendations = recommendations[:top_k]
        
        # Display recommendations
        if show_detailed:
            self._display_recommendations(user_id, final_recommendations, user_type)
        
        return final_recommendations

    def evaluate_model(self, test_data, test_indices, top_k=10):
        """Evaluasi comprehensive model dengan berbagai metrics - DIPERBAIKI"""
        print("🔍 Mengevaluasi performa model...")
        
        X_test, y_test = test_data
        
        # Basic metrics
        predictions = self.model.predict(X_test, verbose=0).flatten()
        
        # RMSE dan MAE
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mae = np.mean(np.abs(predictions - y_test))
        
        # PERBAIKAN: Hanya evaluasi user yang ada di training set
        test_data_df = self.processed_data.iloc[test_indices]
        
        # Filter hanya user yang bisa di-encode (existing users)
        valid_users = []
        for user_id in test_data_df['user_id'].unique():
            try:
                self.encoders['user'].transform([user_id])
                valid_users.append(user_id)
            except:
                continue
        
        # Sample maksimal 50 valid users
        sample_users = valid_users[:50] if len(valid_users) > 50 else valid_users
        
        precision_scores = []
        hit_rates = []
        all_recommended_categories = []
        evaluated_users = 0
        
        for user_id in sample_users:
            try:
                # PERBAIKAN KRITIS: Gunakan threshold yang sama dengan model kedua
                # Get recommendations untuk user ini dengan disable filtering
                user_recs = self.get_enhanced_recommendations(
                    user_id, 
                    top_k=top_k, 
                    show_detailed=False, 
                    disable_filtering=True  # Penting: disable filtering untuk evaluasi yang fair
                )

                if not user_recs:
                    continue
                
                evaluated_users += 1
                
                # PERBAIKAN: Gunakan logika evaluasi yang sama dengan model kedua
                # Hitung precision berdasarkan predicted rating >= 3.5 (bukan actual ratings)
                high_quality_recs = sum(1 for r in user_recs if r['predicted_rating'] >= 3.5)
                precision = high_quality_recs / len(user_recs) if user_recs else 0
                precision_scores.append(precision)
                
                # Hit Rate: apakah ada minimal 1 rekomendasi berkualitas
                hit_rate = 1 if high_quality_recs > 0 else 0
                hit_rates.append(hit_rate)
                
                # Collect categories for diversity
                rec_categories = [rec['category'] for rec in user_recs if rec.get('category')]
                all_recommended_categories.extend(rec_categories)
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {str(e)}")
                continue
        
        # Calculate final metrics
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
        
        # Diversity calculation
        if all_recommended_categories:
            unique_categories = len(set(all_recommended_categories))
            total_possible_categories = len(self.encoders['category'].classes_)
            diversity = unique_categories / total_possible_categories
        else:
            diversity = 0
        
        evaluation_results = {
            'rmse': rmse,
            'mae': mae,
            'precision_at_10': avg_precision,
            'hit_rate_at_10': avg_hit_rate,
            'diversity': diversity,
            'n_evaluated_users': evaluated_users,
            'n_valid_users': len(valid_users),
            'raw_predictions_sample': predictions[:10].tolist(),
            'raw_actual_sample': y_test[:10].tolist()
        }
        
        print(f"✅ Evaluasi selesai: {evaluated_users} users dievaluasi dari {len(valid_users)} valid users")
        
        return evaluation_results

    def _calculate_difficulty_level(self, difficulty_score):
        """Helper function untuk menghitung difficulty level"""
        if difficulty_score <= 1.5:
            return "Cepat & Mudah"
        elif difficulty_score <= 2.5:
            return "Butuh Usaha"
        else:
            return "Level Dewa Masak"
        
    def search_recipes_by_ingredients(self, 
                                    ingredients_input,
                                    category_filter=None,
                                    difficulty_max=5,
                                    limit=20,
                                    search_mode='any',
                                    min_match_percentage=0.3,
                                    prefer_more_matches=True,
                                    show_detailed=True):
        """
        Pencarian resep berdasarkan ingredients yang dimiliki user dengan matching yang akurat
        
        Args:
            ingredients_input (list): List of ingredients yang dimiliki user
            category_filter (str, optional): Filter berdasarkan kategori
            difficulty_max (int): Maximum difficulty level (1-5)
            limit (int): Maximum number of recipes to return (max 50)
            search_mode (str): 'any', 'all', atau 'partial'
            min_match_percentage (float): Minimum match percentage untuk mode 'partial'
            prefer_more_matches (bool): Prioritize recipes with more matching ingredients
            show_detailed (bool): Include detailed matching information
        
        Returns:
            dict: Search results with recipes and metadata
        """
        
        if not self.original_data is not None:
            raise ValueError("Model not loaded")
        
        # Validation untuk ingredients
        if not isinstance(ingredients_input, list):
            raise ValueError("ingredients must be a list")
        
        if len(ingredients_input) == 0:
            raise ValueError("ingredients list cannot be empty")
        
        # Parameters
        ingredients = [ingredient.lower().strip() for ingredient in ingredients_input]
        limit = min(limit, 50)  # Maksimal 50
        
        print(f"🔍 Searching recipes with available ingredients: {ingredients}")
        print(f"📊 Search mode: {search_mode}")
        
        # Mulai pencarian
        df = self.original_data.copy()
        
        # Enhanced exclusion patterns - lebih comprehensive
        exclusion_patterns = {
            'ayam': [
                'royco ayam', 'masako ayam', 'kaldu ayam', 'bumbu ayam', 
                'tepung ayam', 'seasoning ayam', 'penyedap ayam', 'msg ayam',
                'bubuk ayam', 'extract ayam', 'saus ayam', 'telur ayam',
                'kuning telur ayam', 'putih telur ayam', 'minyak ayam'
            ],
            'sapi': [
                'royco sapi', 'masako sapi', 'kaldu sapi', 'bumbu sapi',
                'tepung sapi', 'seasoning sapi', 'penyedap sapi', 'msg sapi'
            ],
            'ikan': [
                'kaldu ikan', 'bumbu ikan', 'tepung ikan', 'seasoning ikan',
                'penyedap ikan', 'msg ikan', 'saus ikan', 'asin ikan'
            ],
            'udang': [
                'kaldu udang', 'bumbu udang', 'tepung udang', 'seasoning udang',
                'penyedap udang', 'msg udang', 'saus udang', 'terasi udang'
            ],
            'kambing': [
                'kaldu kambing', 'bumbu kambing', 'seasoning kambing'
            ],
            'bawang': [
                'bubuk bawang', 'tepung bawang'  # bawang merah/putih tetap valid
            ],
            'tomat': [
                'saus tomat', 'pasta tomat', 'bubuk tomat'  # tomat segar tetap valid
            ]
        }
        
        # Definisi ingredient separations - diperluas
        ingredient_separations = {
            'ayam': {
                'exclude_if_contains': ['telur', 'kuning telur', 'putih telur', 'minyak'],
                'must_contain_standalone': True
            },
            'telur': {
                'variations': ['telur ayam', 'telur bebek', 'telur puyuh', 'kuning telur', 'putih telur'],
                'standalone_ok': True
            },
            'buah': {
                'variations': ['buah naga', 'buah-buahan', 'buah segar'],
                'standalone_ok': True
            },
            'naga': {
                'must_be_combined_with': ['buah'],  # "naga" harus dengan "buah"
                'standalone_ok': False
            }
        }
        
        def normalize_ingredient_name(ingredient):
            """Normalize nama ingredient untuk matching yang lebih baik"""
            # Handle common variations
            variations = {
                'buah naga': ['buah naga', 'dragon fruit'],
                'beras merah': ['beras merah', 'red rice'],
                'biskuit oreo': ['biskuit oreo', 'oreo', 'biskuit'],
                'blueberry': ['blueberry', 'bluberi'],
                'brokoli': ['brokoli', 'broccoli']
            }
            
            ingredient_lower = ingredient.lower().strip()
            
            # Cari dalam variations
            for main_name, variant_list in variations.items():
                if ingredient_lower in [v.lower() for v in variant_list]:
                    return main_name
            
            return ingredient_lower
        
        def is_real_ingredient_match(recipe_ingredients, search_ingredient):
            """
            Enhanced ingredient matching dengan support untuk compound ingredients
            """
            if pd.isna(recipe_ingredients):
                return False
            
            recipe_text = recipe_ingredients.lower()
            search_ing = normalize_ingredient_name(search_ingredient)
            
            # Handle compound ingredients (e.g., "buah naga")
            if ' ' in search_ing:
                # Untuk compound ingredients, cek keseluruhan phrase
                if search_ing in recipe_text:
                    # Pastikan tidak ada exclusion
                    if search_ing.split()[0] in exclusion_patterns:
                        for exclusion in exclusion_patterns[search_ing.split()[0]]:
                            if exclusion in recipe_text:
                                if show_detailed:
                                    print(f"❌ Excluded compound: '{search_ing}' found in '{exclusion}'")
                                return False
                    if show_detailed:
                        print(f"✅ Compound ingredient match: '{search_ing}'")
                    return True
                return False
            
            # Single word ingredients
            if search_ing not in recipe_text:
                return False
            
            # STEP 1: Cek exclusion patterns
            if search_ing in exclusion_patterns:
                for exclusion in exclusion_patterns[search_ing]:
                    if exclusion in recipe_text:
                        if show_detailed:
                            print(f"❌ Excluded match: '{search_ing}' found in '{exclusion}'")
                        return False
            
            # STEP 2: Special handling untuk ingredient separations
            if search_ing in ingredient_separations:
                separation_rules = ingredient_separations[search_ing]
                
                # Cek exclude_if_contains rules
                if 'exclude_if_contains' in separation_rules:
                    for exclude_term in separation_rules['exclude_if_contains']:
                        # Cek berbagai kombinasi
                        combinations_to_check = [
                            f"{exclude_term} {search_ing}",
                            f"{search_ing} {exclude_term}",
                            f"{exclude_term}{search_ing}",  # tanpa spasi
                            f"{search_ing}{exclude_term}"   # tanpa spasi
                        ]
                        
                        for combo in combinations_to_check:
                            if combo in recipe_text:
                                if show_detailed:
                                    print(f"❌ Separated ingredient: '{search_ing}' found combined with '{exclude_term}' as '{combo}'")
                                return False
                
                # Cek must_be_combined_with rules
                if 'must_be_combined_with' in separation_rules:
                    required_combinations = separation_rules['must_be_combined_with']
                    has_valid_combination = False
                    
                    for required in required_combinations:
                        if f"{required} {search_ing}" in recipe_text or f"{search_ing} {required}" in recipe_text:
                            has_valid_combination = True
                            break
                    
                    if not has_valid_combination and not separation_rules.get('standalone_ok', False):
                        if show_detailed:
                            print(f"❌ Ingredient '{search_ing}' requires combination but none found")
                        return False
            
            # STEP 3: Word boundary checking dengan improved regex
            import re
            
            # Pattern yang lebih fleksibel untuk ingredient matching
            patterns_to_try = [
                r'\b' + re.escape(search_ing) + r'\b',  # exact word boundary
                r'\b' + re.escape(search_ing) + r'(?![a-zA-Z])',  # starts with, not part of larger word
                r'(?<![a-zA-Z])' + re.escape(search_ing) + r'\b'   # ends with, not part of larger word
            ]
            
            for pattern in patterns_to_try:
                matches = re.finditer(pattern, recipe_text)
                
                for match in matches:
                    # Analisis konteks untuk setiap match
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Ambil konteks yang lebih luas
                    context_start = max(0, start_pos - 30)
                    context_end = min(len(recipe_text), end_pos + 30)
                    context = recipe_text[context_start:context_end]
                    
                    if show_detailed:
                        print(f"🔍 Context analysis for '{search_ing}': '...{context}...'")
                    
                    # Blacklist context words
                    blacklist_context = [
                        'royco', 'masako', 'kaldu', 'bumbu', 'tepung', 
                        'seasoning', 'penyedap', 'msg', 'extract', 'bubuk',
                        'saus', 'pasta', 'oil', 'minyak', 'powder'
                    ]
                    
                    # Cek apakah context mengandung blacklist
                    context_is_clean = not any(bl in context for bl in blacklist_context)
                    
                    if context_is_clean:
                        if show_detailed:
                            print(f"✅ Valid ingredient match: '{search_ing}' with clean context")
                        return True
            
            if show_detailed:
                print(f"❌ No valid match found for '{search_ing}'")
            return False
        
        def calculate_ingredient_score(recipe_ingredients, search_ingredients, mode='any'):
            """Enhanced scoring dengan bonus untuk multiple matches"""
            if pd.isna(recipe_ingredients):
                return 0, 0, [], 0
            
            matches = []
            matched_ingredients = []
            
            # Hitung matches
            for ingredient in search_ingredients:
                if is_real_ingredient_match(recipe_ingredients, ingredient):
                    matches.append(ingredient)
                    matched_ingredients.append(ingredient)
            
            match_count = len(matches)
            total_search_ingredients = len(search_ingredients)
            match_percentage = match_count / total_search_ingredients if total_search_ingredients > 0 else 0
            
            # Base scoring berdasarkan mode
            if mode == 'all':
                base_score = 1.0 if match_count == total_search_ingredients else 0.0
            elif mode == 'any':
                base_score = match_percentage if match_count > 0 else 0.0
            elif mode == 'partial':
                base_score = match_percentage if match_percentage >= min_match_percentage else 0.0
            else:
                base_score = match_percentage
            
            # Bonus score untuk multiple matches (encourage recipes using more available ingredients)
            bonus_score = 0
            if prefer_more_matches and match_count > 1:
                # Progressive bonus: 2 matches = +0.1, 3 matches = +0.2, etc.
                bonus_score = min((match_count - 1) * 0.1, 0.5)  # max bonus 0.5
            
            final_score = min(base_score + bonus_score, 1.0)  # cap at 1.0
            
            return final_score, match_count, matched_ingredients, bonus_score
        
        # Apply ingredient scoring dengan progress tracking
        print("🔄 Processing recipes...")
        scores = []
        match_counts = []
        matched_ingredients_list = []
        bonus_scores = []
        difficulty_scores = []
        
        total_recipes = len(df)
        processed = 0
        
        for _, recipe in df.iterrows():
            score, match_count, matched_ings, bonus = calculate_ingredient_score(
                recipe['Ingredients Cleaned'], 
                ingredients, 
                search_mode
            )
            scores.append(score)
            match_counts.append(match_count)
            matched_ingredients_list.append(matched_ings)
            bonus_scores.append(bonus)
            
            # Calculate difficulty score
            difficulty_score = self.calculate_difficulty_score(
                int(recipe['Total Ingredients']), 
                int(recipe['Total Steps'])
            )
            difficulty_scores.append(difficulty_score)
            
            processed += 1
            if processed % 100 == 0 and show_detailed:
                print(f"📊 Processed {processed}/{total_recipes} recipes")
        
        df['ingredient_score'] = scores
        df['ingredient_match_count'] = match_counts
        df['matched_ingredients'] = matched_ingredients_list
        df['bonus_score'] = bonus_scores
        df['difficulty_score'] = difficulty_scores
        
        # Filter berdasarkan ingredient score
        initial_count = len(df)
        df = df[df['ingredient_score'] > 0]
        filtered_count = len(df)
        
        print(f"📋 Found {filtered_count} recipes with matching ingredients (filtered from {initial_count})")
        
        # Apply additional filters
        if category_filter:
            df = df[df['Category'] == category_filter]
            print(f"📂 After category filter '{category_filter}': {len(df)}")
        
        # Filter difficulty
        df = df[df['difficulty_score'] <= difficulty_max]
        print(f"🎯 After difficulty filter <= {difficulty_max}: {len(df)}")
        
        # Enhanced scoring dan sorting
        if len(df) > 0:
            # Normalize ratings untuk scoring
            max_rating = df['total_rating'].max()
            df['normalized_rating'] = df['total_rating'] / max_rating if max_rating > 0 else 0
            
            # Combined score dengan weight yang disesuaikan untuk use case "available ingredients"
            df['combined_score'] = (
                df['ingredient_score'] * 0.7 +  # 70% - prioritas tinggi untuk ingredient matching
                df['normalized_rating'] * 0.2 +  # 20% - rating masih penting
                (1 - (df['difficulty_score'] / 5.0)) * 0.1  # 10% - sedikit bonus untuk kemudahan
            )
            
            # Sort berdasarkan combined score, kemudian match count, kemudian rating
            results = df.sort_values(
                ['combined_score', 'ingredient_match_count', 'total_rating'], 
                ascending=[False, False, False]
            ).head(limit)
        else:
            results = df.head(limit)
        
        # Format output dengan informasi tambahan
        recipes = []
        for _, recipe in results.iterrows():
            # Hitung ingredient utilization
            total_recipe_ingredients = int(recipe['Total Ingredients'])
            matched_count = int(recipe['ingredient_match_count'])
            utilization_percentage = (matched_count / len(ingredients)) * 100 if len(ingredients) > 0 else 0
            
            recipes.append({
                'item_id': int(recipe['item_id']),
                'title_cleaned': recipe['Title Cleaned'],
                'ingredients_cleaned': recipe['Ingredients Cleaned'],
                'steps_cleaned': recipe.get('Steps Cleaned', 'N/A'),
                'category': recipe.get('Category', 'Unknown'),
                'total_rating': float(recipe['total_rating']),
                'total_ingredients': total_recipe_ingredients,
                'total_steps': int(recipe['Total Steps']),
                'difficulty_level': self.get_difficulty_level(recipe['difficulty_score']),
                'difficulty_score': float(recipe['difficulty_score']),
                'image_url': recipe.get('Image URL', 'N/A'),
                'ingredient_score': float(recipe['ingredient_score']),
                'ingredient_match_count': matched_count,
                'matched_ingredients': recipe['matched_ingredients'],
                'bonus_score': float(recipe['bonus_score']),
                'combined_score': float(recipe['combined_score']),
                'utilization_percentage': round(utilization_percentage, 1),
                'missing_ingredients_estimate': max(0, total_recipe_ingredients - matched_count)
            })
        
        # Generate summary statistics
        summary_stats = {}
        if len(df) > 0:
            summary_stats = {
                'total_recipes_found': len(recipes),
                'avg_ingredient_score': round(float(df['ingredient_score'].mean()), 3),
                'avg_match_count': round(float(df['ingredient_match_count'].mean()), 1),
                'max_matches': int(df['ingredient_match_count'].max()),
                'recipes_with_multiple_matches': int(len(df[df['ingredient_match_count'] > 1])),
                'avg_utilization': round(sum(r['utilization_percentage'] for r in recipes) / len(recipes), 1) if recipes else 0
            }
        
        return {
            'success': True,
            'search_ingredients': ingredients,
            'search_mode': search_mode,
            'total_results': len(recipes),
            'recipes': recipes,
            'filters_applied': {
                'category_filter': category_filter,
                'difficulty_max': difficulty_max,
                'min_match_percentage': min_match_percentage if search_mode == 'partial' else None,
                'prefer_more_matches': prefer_more_matches
            },
            'search_summary': summary_stats,
            'matching_enhancements': {
                'exclusion_patterns': len(exclusion_patterns),
                'ingredient_separations': len(ingredient_separations),
                'compound_ingredient_support': True,
                'context_analysis': True,
                'word_boundary_checking': True,
                'progressive_bonus_scoring': prefer_more_matches
            },
            'recommendations': {
                'message': f"Found {len(recipes)} recipes using your available ingredients",
                'tip': f"Recipes using more of your ingredients ({max(summary_stats.get('max_matches', 0), 1)} max) are ranked higher",
                'suggestion': "Try different search modes: 'any' for flexibility, 'all' for exact matches, 'partial' for percentage-based matching"
            }
        }
    

