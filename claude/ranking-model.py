import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class RankingModel:
    def __init__(self, processed_articles, user_features, article_features):
        self.processed_articles = processed_articles
        self.user_features = user_features
        self.article_features = article_features
        
        # Extract embedding columns
        self.emb_cols = [col for col in processed_articles.columns if col.startswith('emb_')]
        
        # LightGBM model
        self.model = None
    
    def _create_user_article_features(self, user_id, article_id, user_embedding=None):
        """Create features for a user-article pair"""
        features = {}
        
        # User features
        if user_id in self.user_features.index:
            for col in self.user_features.columns:
                features[f'user_{col}'] = self.user_features.loc[user_id, col]
        else:
            # Default values for new users
            for col in self.user_features.columns:
                features[f'user_{col}'] = 0
        
        # Article features
        article_row = self.article_features[self.article_features['article_id'] == article_id]
        if not article_row.empty:
            features['article_popularity'] = article_row['popularity'].values[0]
            features['article_recency'] = article_row['recency_score'].values[0]
            features['article_pop_recency'] = article_row['pop_recency_score'].values[0]
            features['article_words_count'] = article_row['words_count'].values[0]
            
            # Get article category
            article_info = self.processed_articles[self.processed_articles['article_id'] == article_id]
            if not article_info.empty:
                features['article_category'] = article_info['category_id'].values[0]
                
                # Article embedding similarity if user embedding is provided
                if user_embedding is not None:
                    article_emb = article_info[self.emb_cols].values[0]
                    features['embedding_similarity'] = np.dot(user_embedding, article_emb) / (
                        np.linalg.norm(user_embedding) * np.linalg.norm(article_emb)
                    )
                else:
                    features['embedding_similarity'] = 0
        else:
            # Default values for new articles
            features['article_popularity'] = 0
            features['article_recency'] = 0
            features['article_pop_recency'] = 0
            features['article_words_count'] = 0
            features['article_category'] = -1
            features['embedding_similarity'] = 0
            
        return features
    
    def _get_user_embedding(self, user_clicks):
        """Get user embedding from clicked articles"""
        if not user_clicks:
            return None
            
        # Get article embeddings
        article_embeddings = []
        for article_id in user_clicks:
            article_row = self.processed_articles[self.processed_articles['article_id'] == article_id]
            if not article_row.empty:
                emb = article_row[self.emb_cols].values[0]
                article_embeddings.append(emb)
        
        if not article_embeddings:
            return None
            
        # Average article embeddings
        user_emb = np.mean(article_embeddings, axis=0)
        return user_emb
    
    def prepare_training_data(self, processed_clicks):
        """Prepare training data for the ranking model"""
        # Get user click history
        user_clicks = {}
        for user_id, group in processed_clicks.groupby('user_id'):
            sorted_clicks = group.sort_values('click_timestamp')
            user_clicks[user_id] = sorted_clicks['click_article_id'].tolist()
        
        # Create training examples
        X = []
        y = []
        
        for user_id, clicks in user_clicks.items():
            if len(clicks) < 2:
                continue
                
            # Use all clicks except the last one for user embedding
            history = clicks[:-1]
            user_emb = self._get_user_embedding(history)
            
            # Last click is positive example
            pos_article = clicks[-1]
            pos_features = self._create_user_article_features(user_id, pos_article, user_emb)
            X.append(pos_features)
            y.append(1)
            
            # Sample negative examples (articles not clicked by the user)
            all_articles = set(self.processed_articles['article_id'])
            neg_articles = list(all_articles - set(clicks))
            
            # Sample 5 negative examples per positive
            if neg_articles:
                neg_sample = np.random.choice(neg_articles, min(5, len(neg_articles)), replace=False)
                for neg_article in neg_sample:
                    neg_features = self._create_user_article_features(user_id, neg_article, user_emb)
                    X.append(neg_features)
                    y.append(0)
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X)
        y_array = np.array(y)
        
        return X_df, y_array
    
    def train(self, processed_clicks):
        """Train the ranking model"""
        X, y = self.prepare_training_data(processed_clicks)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Categorical columns
        cat_cols = [col for col in X.columns if col.endswith('_category') or 
                    col.endswith('_environment') or col.endswith('_deviceGroup') or 
                    col.endswith('_os') or col.endswith('_country') or 
                    col.endswith('_region') or col.endswith('_referrer_type')]
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Save model
        self.model.save_model('ranking_model.txt')
        
        return self.model
    
    def predict(self, user_id, candidates, user_clicks):
        """Predict scores for candidate articles"""
        if not self.model:
            raise ValueError("Model not trained yet")
            
        # Get user embedding
        user_emb = self._get_user_embedding(user_clicks)
        
        # Create features for all candidates
        candidate_features = []
        for article_id in candidates:
            features = self._create_user_article_features(user_id, article_id, user_emb)
            candidate_features.append(features)
            
        # Convert to DataFrame
        X_pred = pd.DataFrame(candidate_features)
        
        # Predict scores
        scores = self.model.predict(X_pred)
        
        # Combine article IDs with scores
        results = list(zip(candidates, scores))
        
        # Sort by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

# Train the ranking model
ranking_model = RankingModel(processed_articles, user_features, article_features)
trained_model = ranking_model.train(processed_train_clicks)

# Load candidates for test users
with open('test_user_candidates.pkl', 'rb') as f:
    test_user_candidates = pickle.load(f)

# Get predictions for test users
test_predictions = {}
for user_id, candidates in test_user_candidates.items():
    user_clicks = test_user_history.get(user_id, [])
    ranked_candidates = ranking_model.predict(user_id, candidates, user_clicks)
    # Keep only the article IDs, not the scores
    test_predictions[user_id] = [article_id for article_id, _ in ranked_candidates[:5]]
