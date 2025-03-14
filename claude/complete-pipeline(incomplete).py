import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class ArticleRecommender:
    def __init__(self):
        self.data_processor = None
        self.candidate_generator = None
        self.ranking_model = None
        self.article_features = None
        self.user_features = None
        self.processed_articles = None
    
    def train(self, train_clicks_path, articles_path, article_emb_path):
        """Train the full recommendation pipeline"""
        # Step 1: Process data
        print("Processing data...")
        self._process_data(train_clicks_path, articles_path, article_emb_path)
        
        # Step 2: Create candidate generator
        print("Setting up candidate generator...")
        self._setup_candidate_generator()
        
        # Step 3: Train ranking model
        print("Training ranking model...")
        self._train_ranking_model()
        
        print("Training complete!")
        return self
    
    def _process_data(self, train_clicks_path, articles_path, article_emb_path):
        """Process all data"""
        # Load data
        train_clicks = pd.read_csv(train_clicks_path)
        articles = pd.read_csv(articles_path)
        article_embeddings = pd.read_csv(article_emb_path)
        
        # Process articles
        self.processed_articles = articles.merge(article_embeddings, on='article_id', how='left')
        
        # Calculate article age
        min_ts = self.processed_articles['created_at_ts'].min()
        self.processed_articles['article_age'] = (self.processed_articles['created_at_ts'] - min_ts) / (24 * 3600)
        
        # Process clicks
        processed_clicks = train_clicks.merge(
            self.processed_articles[['article_id', 'category_id', 'created_at_ts', 'words_count']], 
            left_on='click_article_id', 
            right_on='article_id', 
            how='left'
        )
        
        # Add time features
        processed_clicks['click_datetime'] = pd.to_datetime(processed_clicks['click_timestamp'], unit='s')
        processed_clicks['hour_of_day'] = processed_clicks['click_datetime'].dt.hour
        processed_clicks['day_of_week'] = processed_clicks['click_datetime'].dt.dayofweek
        processed_clicks['article_age_at_click'] = (processed_clicks['click_timestamp'] - 
                                           processed_clicks['created_at_ts']) / (24 * 3600)
        
        # Create user features
        self.user_features = processed_clicks.groupby('user_id').agg({
            'click_article_