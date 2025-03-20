import pandas as pd
import numpy as np
import xgboost as xgb
import time
import os
import pickle
import multiprocessing
from functools import partial
from tqdm import tqdm
import h5py

def vectorized_cosine_similarity(matrix_a, vector_b):
    """Compute cosine similarity between a matrix and a vector in a vectorized way"""
    # Normalize vector_b
    norm_b = np.linalg.norm(vector_b)
    if norm_b == 0:
        return np.zeros(matrix_a.shape[0])
    vector_b_normalized = vector_b / norm_b
    
    # Normalize each row in matrix_a
    norms_a = np.linalg.norm(matrix_a, axis=1)
    valid_indices = norms_a != 0
    
    similarities = np.zeros(matrix_a.shape[0])
    if valid_indices.any():
        matrix_a_normalized = matrix_a[valid_indices] / norms_a[valid_indices].reshape(-1, 1)
        similarities[valid_indices] = np.dot(matrix_a_normalized, vector_b_normalized)
    
    return similarities

def precompute_user_profiles(train_clicks, articles_df, embed_cols, cache_file='user_profiles.h5'):
    """Precompute and cache user profiles for all users"""
    if os.path.exists(cache_file):
        print(f"Loading precomputed user profiles from {cache_file}")
        with h5py.File(cache_file, 'r') as f:
            user_ids = f['user_ids'][:]
            profiles = f['profiles'][:]
            return dict(zip(user_ids, profiles))
    
    print("Precomputing user profiles...")
    
    # Create a mapping from article_id to its embedding
    article_id_to_idx = {id_: i for i, id_ in enumerate(articles_df['article_id'])}
    article_embeddings = articles_df[embed_cols].values
    
    # Group clicks by user_id
    user_groups = train_clicks.groupby('user_id')
    user_profiles = {}
    
    for uid, group in tqdm(user_groups):
        # Sort by timestamp
        group = group.sort_values('click_timestamp')
        if len(group) < 2:
            continue
        
        # Get history (all but the last click)
        history = group.iloc[:-1]
        
        # Get article indices for the articles in the user's history
        article_indices = [article_id_to_idx.get(aid, -1) for aid in history['click_article_id']]
        valid_indices = [i for i in article_indices if i != -1]
        
        if not valid_indices:
            continue
        
        # Compute the mean embedding directly from the embeddings matrix
        profile = article_embeddings[valid_indices].mean(axis=0)
        user_profiles[uid] = profile
    
    # Cache the results
    print(f"Saving user profiles to {cache_file}")
    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('user_ids', data=np.array(list(user_profiles.keys())))
        f.create_dataset('profiles', data=np.array(list(user_profiles.values())))
    
    return user_profiles

def process_user_batch(user_batch, train_clicks, articles_df, embed_cols, user_profiles, article_id_to_idx):
    """Process a batch of users to generate training data"""
    article_embeddings = articles_df[embed_cols].values
    batch_train_data = []
    batch_train_labels = []
    batch_group_ptr = []
    
    for uid in user_batch:
        group = train_clicks[train_clicks['user_id'] == uid].sort_values('click_timestamp')
        if len(group) < 2:
            continue
        
        # Get the positive example (last click)
        pos_click = group.iloc[-1]
        pos_article_id = pos_click['click_article_id']
        
        # Get user profile from precomputed dictionary
        user_profile = user_profiles.get(uid)
        if user_profile is None:
            continue
        
        # Get positive article embedding
        pos_idx = article_id_to_idx.get(pos_article_id, -1)
        if pos_idx == -1:
            continue
        pos_embedding = article_embeddings[pos_idx]
        
        # Compute positive score
        pos_score = np.dot(user_profile / np.linalg.norm(user_profile), 
                           pos_embedding / np.linalg.norm(pos_embedding))
        
        batch_train_data.append([pos_score])
        batch_train_labels.append(1)
        
        # Generate negative examples using fixed random seed for reproducibility
        n_negatives = 4
        np.random.seed(42 + uid)  # Use user_id to vary the seed but keep it deterministic
        neg_indices = np.random.choice(len(articles_df), n_negatives, replace=False)
        
        for neg_idx in neg_indices:
            neg_embedding = article_embeddings[neg_idx]
            # Compute negative score
            neg_norm = np.linalg.norm(neg_embedding)
            if neg_norm == 0:
                neg_score = 0
            else:
                neg_score = np.dot(user_profile / np.linalg.norm(user_profile), 
                                neg_embedding / neg_norm)
            
            batch_train_data.append([neg_score])
            batch_train_labels.append(0)
        
        batch_group_ptr.append(1 + n_negatives)
    
    return batch_train_data, batch_train_labels, batch_group_ptr

def prepare_training_data(train_clicks, articles_df, embed_cols, cache_file):
    """Prepare training data with caching"""
    if os.path.exists(cache_file):
        print(f"Loading precomputed training data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    start_time = time.time()
    print("Preparing training data...")
    
    # Precompute user profiles
    user_profiles = precompute_user_profiles(train_clicks, articles_df, embed_cols)
    print(f"Precomputed {len(user_profiles)} user profiles in {time.time() - start_time:.2f} seconds")
    
    # Create article ID to index mapping
    article_id_to_idx = {id_: i for i, id_ in enumerate(articles_df['article_id'])}
    
    # Get unique user IDs
    unique_user_ids = list(user_profiles.keys())
    
    # Split users into batches for parallel processing
    num_cores = multiprocessing.cpu_count()
    batch_size = max(1, len(unique_user_ids) // (num_cores * 2))
    user_batches = [unique_user_ids[i:i+batch_size] for i in range(0, len(unique_user_ids), batch_size)]
    
    print(f"Processing {len(unique_user_ids)} users in {len(user_batches)} batches using {num_cores} cores")
    
    # Process batches in parallel
    process_batch_fn = partial(
        process_user_batch,
        train_clicks=train_clicks,
        articles_df=articles_df,
        embed_cols=embed_cols,
        user_profiles=user_profiles,
        article_id_to_idx=article_id_to_idx
    )
    
    # Use multiprocessing pool to process batches
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_batch_fn, user_batches), total=len(user_batches)))
    
    # Combine results from all batches
    train_data = []
    train_labels = []
    group_ptr = []
    
    for batch_data, batch_labels, batch_ptr in results:
        train_data.extend(batch_data)
        train_labels.extend(batch_labels)
        group_ptr.extend(batch_ptr)
    
    # Cache the results
    result = (np.array(train_data), np.array(train_labels), group_ptr)
    print(f"Saving training data to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Training data preparation completed in {time.time() - start_time:.2f} seconds")
    return result

# Usage example
if __name__ == "__main__":
    start_time = time.time()
    colab = True
    prefix = 'drive/MyDrive/' if colab else ''
    # Load data
    articles = pd.read_csv(prefix + 'news/articles.csv')
    embed_cols = articles.columns[4:]
    train_clicks = pd.read_csv(prefix + 'news/train_click_log.csv')
    
    print(f"Read CSV files in {time.time() - start_time:.2f} seconds")
    
    # Prepare training data with caching
    train_data, train_labels, group_ptr = prepare_training_data(train_clicks, articles, embed_cols, prefix + 'training_data.pkl')
    
    # Create XGBoost DMatrix
    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dtrain.set_group(group_ptr)
    
    # Set parameters
    params = {
        'objective': 'rank:pairwise',
        'eta': 0.1,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'max_depth': 6,
        'tree_method': 'hist',
        'device': 'cuda',
        'verbosity': 1
    }
    
    # Train model
    print("Training XGBoost model...")
    model_start = time.time()
    num_round = 50
    model = xgb.train(params, dtrain, num_boost_round=num_round)
    print(f"Model training completed in {time.time() - model_start:.2f} seconds")
    
    # Save model
    model.save_model(prefix + 'news_rec_model.json')
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

# Read CSV files in 19.43 seconds
# Preparing training data...
# Precomputing user profiles...
# 100%|██████████| 200000/200000 [00:44<00:00, 4468.38it/s]
# Saving user profiles to user_profiles.h5
# Precomputed 200000 user profiles in 47.61 seconds
# Processing 200000 users in 16 batches using 8 cores
# 100%|██████████| 16/16 [07:18<00:00, 27.39s/it]
# Saving training data to drive/MyDrive/training_data.pkl
# Training data preparation completed in 489.32 seconds
# Training XGBoost model...
# Model training completed in 17.47 seconds
# Total processing time: 526.84 seconds