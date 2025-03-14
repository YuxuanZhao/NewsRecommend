import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load the datasets
train_clicks = pd.read_csv('train_click_log.csv')
test_clicks = pd.read_csv('test_click_log.csv')
articles = pd.read_csv('articles.csv')
article_embeddings = pd.read_csv('articles_emb.csv')

# Process articles data
def process_articles(articles_df, embeddings_df):
    # Merge with embeddings
    articles_full = articles_df.merge(embeddings_df, on='article_id', how='left')
    
    # Calculate article age in days (relative to the oldest article)
    min_ts = articles_full['created_at_ts'].min()
    articles_full['article_age'] = (articles_full['created_at_ts'] - min_ts) / (24 * 3600)
    
    # Create category dummies for faster operations later
    articles_full['category_id'] = articles_full['category_id'].astype('category')
    
    return articles_full

# Process click logs
def process_clicks(clicks_df, articles_df):
    # Merge with article info
    clicks_with_info = clicks_df.merge(
        articles_df[['article_id', 'category_id', 'created_at_ts', 'words_count']], 
        left_on='click_article_id', 
        right_on='article_id', 
        how='left'
    )
    
    # Convert timestamp to datetime for time-based features
    clicks_with_info['click_datetime'] = pd.to_datetime(clicks_with_info['click_timestamp'], unit='s')
    clicks_with_info['hour_of_day'] = clicks_with_info['click_datetime'].dt.hour
    clicks_with_info['day_of_week'] = clicks_with_info['click_datetime'].dt.dayofweek
    
    # Article age at time of click
    clicks_with_info['article_age_at_click'] = (clicks_with_info['click_timestamp'] - 
                                               clicks_with_info['created_at_ts']) / (24 * 3600)
    
    # Convert categorical features to category type
    cat_cols = ['click_environment', 'click_deviceGroup', 'click_os', 
                'click_country', 'click_region', 'click_referrer_type']
    for col in cat_cols:
        clicks_with_info[col] = clicks_with_info[col].astype('category')
    
    return clicks_with_info

# Extract user features (behavior patterns)
def create_user_features(clicks_df):
    user_features = clicks_df.groupby('user_id').agg({
        'click_article_id': ['count', 'nunique'],
        'category_id': lambda x: x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else -1,
        'click_environment': lambda x: x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else -1,
        'click_deviceGroup': lambda x: x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else -1,
        'click_os': lambda x: x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else -1,
        'words_count': ['mean', 'std'],
        'hour_of_day': ['mean', 'std'],
        'article_age_at_click': ['mean', 'std']
    })
    
    # Flatten multi-level columns
    user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
    
    # Normalize numerical features
    scaler = StandardScaler()
    num_cols = ['click_article_id_count', 'click_article_id_nunique', 
                'words_count_mean', 'words_count_std', 
                'hour_of_day_mean', 'hour_of_day_std',
                'article_age_at_click_mean', 'article_age_at_click_std']
    
    for col in num_cols:
        if col in user_features.columns:
            user_features[col] = scaler.fit_transform(user_features[[col]].fillna(0))
    
    return user_features

# Create article popularity features
def create_article_features(clicks_df, articles_df):
    # Count clicks per article
    article_pop = clicks_df.groupby('click_article_id').size().reset_index(name='click_count')
    
    # Merge with article info
    article_features = articles_df.merge(article_pop, left_on='article_id', right_on='click_article_id', how='left')
    article_features['click_count'] = article_features['click_count'].fillna(0)
    
    # Calculate recency score (higher for newer articles)
    max_ts = clicks_df['click_timestamp'].max()
    article_features['recency_score'] = 1 / (1 + np.log1p((max_ts - article_features['created_at_ts']) / (24 * 3600)))
    
    # Normalize click count (popularity)
    article_features['popularity'] = article_features['click_count'] / article_features['click_count'].max()
    
    # Combine popularity and recency
    article_features['pop_recency_score'] = 0.7 * article_features['popularity'] + 0.3 * article_features['recency_score']
    
    return article_features

# Create user-category preference matrix
def create_user_category_matrix(clicks_df):
    user_cat_counts = clicks_df.groupby(['user_id', 'category_id']).size().reset_index(name='count')
    
    # Convert to user-category matrix
    user_cat_matrix = user_cat_counts.pivot(index='user_id', columns='category_id', values='count')
    user_cat_matrix = user_cat_matrix.fillna(0)
    
    # Normalize by user
    user_totals = user_cat_matrix.sum(axis=1)
    user_cat_matrix = user_cat_matrix.div(user_totals, axis=0)
    
    return user_cat_matrix

# Process all data
processed_articles = process_articles(articles, article_embeddings)
processed_train_clicks = process_clicks(train_clicks, processed_articles)
processed_test_clicks = process_clicks(test_clicks, processed_articles)

# Create features
user_features = create_user_features(processed_train_clicks)
article_features = create_article_features(processed_train_clicks, processed_articles)
user_category_matrix = create_user_category_matrix(processed_train_clicks)

# Get the last click for each test user (ground truth)
test_users_last_clicks = processed_test_clicks.groupby('user_id').apply(
    lambda x: x.loc[x['click_timestamp'].idxmax()]
).reset_index(drop=True)[['user_id', 'click_article_id']]
