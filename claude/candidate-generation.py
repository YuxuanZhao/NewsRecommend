import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import pickle

class CandidateGenerator:
    def __init__(self, processed_articles, article_features, user_category_matrix):
        self.processed_articles = processed_articles
        self.article_features = article_features
        self.user_category_matrix = user_category_matrix
        self.article_ids = processed_articles['article_id'].unique()
        
        # Extract article embeddings
        self.article_id_to_idx = {id: i for i, id in enumerate(processed_articles['article_id'])}
        self.idx_to_article_id = {i: id for id, i in self.article_id_to_idx.items()}
        
        # Prepare embedding matrix
        emb_cols = [col for col in processed_articles.columns if col.startswith('emb_')]
        self.article_embeddings = processed_articles[emb_cols].values
        
        # Prepare popularity items
        self.popular_items = article_features.sort_values('pop_recency_score', ascending=False)[
            'article_id'].values[:1000]
    
    def generate_popular_candidates(self, n=500):
        """Return most popular articles as candidates"""
        return self.popular_items[:n]
    
    def generate_category_candidates(self, user_id, n=200):
        """Generate candidates based on user's category preferences"""
        if user_id not in self.user_category_matrix.index:
            return []
        
        # Get user's preferred categories
        user_cats = self.user_category_matrix.loc[user_id]
        top_cats = user_cats.sort_values(ascending=False).index[:3]
        
        # Get articles from these categories
        cat_articles = []
        for cat in top_cats:
            cat_arts = self.processed_articles[self.processed_articles['category_id'] == cat]['article_id'].values
            cat_articles.extend(cat_arts)
        
        # Sort by popularity within category
        cat_article_scores = []
        for art_id in cat_articles:
            if art_id in self.article_id_to_idx:
                score = self.article_features.loc[
                    self.article_features['article_id'] == art_id, 'pop_recency_score'
                ].values[0]
                cat_article_scores.append((art_id, score))
        
        cat_article_scores.sort(key=lambda x: x[1], reverse=True)
        return [art_id for art_id, _ in cat_article_scores[:n]]
    
    def get_user_embedding(self, user_clicks):
        """Create a user embedding based on average of clicked article embeddings"""
        if len(user_clicks) == 0:
            return None
        
        # Get embeddings for articles user has clicked
        article_indices = [self.article_id_to_idx[art_id] for art_id in user_clicks 
                          if art_id in self.article_id_to_idx]
        
        if not article_indices:
            return None
            
        # Calculate average embedding
        clicked_embeddings = self.article_embeddings[article_indices]
        user_emb = np.mean(clicked_embeddings, axis=0)
        return user_emb
    
    def generate_embedding_candidates(self, user_embedding, n=300):
        """Generate candidates similar to user's embedding"""
        if user_embedding is None:
            return []
            
        # Calculate similarity to all articles
        similarities = cosine_similarity([user_embedding], self.article_embeddings)[0]
        
        # Get top N articles by similarity
        top_indices = np.argsort(similarities)[-n:][::-1]
        candidates = [self.idx_to_article_id[idx] for idx in top_indices]
        return candidates
    
    def generate_candidates(self, user_id, user_clicks):
        """Generate combined candidates for a user"""
        candidates = set()
        
        # Add popular items
        candidates.update(self.generate_popular_candidates(n=200))
        
        # Add category-based items
        if user_id in self.user_category_matrix.index:
            candidates.update(self.generate_category_candidates(user_id, n=200))
        
        # Add embedding-based items
        user_embedding = self.get_user_embedding(user_clicks)
        if user_embedding is not None:
            candidates.update(self.generate_embedding_candidates(user_embedding, n=300))
        
        return list(candidates)

# Function to create click history for test users based on test_clicks
def get_test_click_history(test_clicks):
    user_history = defaultdict(list)
    
    # Group by user and sort by timestamp
    for user_id, group in test_clicks.groupby('user_id'):
        sorted_clicks = group.sort_values('click_timestamp')
        
        # Get all but the last click
        history = sorted_clicks['click_article_id'].values[:-1]
        user_history[user_id] = history.tolist()
    
    return user_history

# Set up candidate generator
candidate_generator = CandidateGenerator(processed_articles, article_features, user_category_matrix)

# Get click history for test users (excluding the last click which is ground truth)
test_user_history = get_test_click_history(processed_test_clicks)

# Generate candidates for each test user
test_user_candidates = {}
for user_id, click_history in test_user_history.items():
    candidates = candidate_generator.generate_candidates(user_id, click_history)
    test_user_candidates[user_id] = candidates

# Save the candidates for later use in ranking
with open('test_user_candidates.pkl', 'wb') as f:
    pickle.dump(test_user_candidates, f)
