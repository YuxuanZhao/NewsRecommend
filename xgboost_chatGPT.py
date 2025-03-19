import pandas as pd
import numpy as np
import xgboost as xgb
import math
import faiss
import time

articles = pd.read_csv('drive/MyDrive/news/articles.csv')
embed_cols = articles.columns[4:]
train_clicks = pd.read_csv('drive/MyDrive/news/train_click_log.csv')
test_clicks = pd.read_csv('drive/MyDrive/news/test_click_log.csv')
train_clicks = train_clicks.head(100000) # 1112623
test_clicks = test_clicks.head(10000) # 518010

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0
    return np.dot(u, v) / (norm_u * norm_v)

def build_user_profile(clicks, articles_df):
    merged = clicks.merge(articles_df, left_on='click_article_id', right_on='article_id', how='left')
    profile = merged[embed_cols].mean().values
    return profile

def compute_ndcg(rank):
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def vectorized_cosine_similarity_cpu(user_profile, article_embeddings):
    """
    Compute cosine similarity between a user profile and all article embeddings at once using CPU
    """
    # Normalize user profile
    user_norm = np.linalg.norm(user_profile)
    if user_norm == 0:
        return np.zeros(article_embeddings.shape[0])
    user_profile_normalized = user_profile / user_norm
    
    # Normalize article embeddings
    article_norms = np.linalg.norm(article_embeddings, axis=1)
    valid_indices = article_norms != 0
    
    # Initialize scores array
    scores = np.zeros(article_embeddings.shape[0])
    
    # Compute normalized embeddings where valid
    article_embeddings_normalized = np.zeros_like(article_embeddings)
    article_embeddings_normalized[valid_indices] = article_embeddings[valid_indices] / article_norms[valid_indices].reshape(-1, 1)
    
    # Compute dot products (equivalent to cosine similarity with normalized vectors)
    scores = np.dot(article_embeddings_normalized, user_profile_normalized)
    
    return scores

def batch_predict_for_users(user_groups, articles_df, model, batch_size=32):
    """
    Process predictions for multiple users in batches using CPU
    """
    all_results = []
    
    # Pre-compute article embeddings matrix once (n_articles x embed_dim)
    article_embeddings = articles_df[embed_cols].values
    
    # Normalize article embeddings for faster similarity computation
    article_norms = np.linalg.norm(article_embeddings, axis=1)
    valid_indices = article_norms != 0
    article_embeddings_normalized = np.zeros_like(article_embeddings)
    article_embeddings_normalized[valid_indices] = article_embeddings[valid_indices] / article_norms[valid_indices].reshape(-1, 1)
    
    # Create CPU index with Faiss
    d = article_embeddings.shape[1]  # dimension
    index = faiss.IndexFlatIP(d)  # Inner product is equivalent to cosine similarity for normalized vectors
    index.add(article_embeddings_normalized.astype(np.float32))  # Add normalized vectors
    
    # Process users in batches
    user_ids = list(user_groups.groups.keys())
    
    for i in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_results = []
        
        for uid in batch_user_ids:
            group = user_groups.get_group(uid)
            if len(group) < 2:
                continue  # not enough history
                
            group_sorted = group.sort_values('click_timestamp')
            ground_truth = group_sorted.iloc[-1]['click_article_id']
            
            # Get user profile
            history = group_sorted.iloc[:-1]
            user_profile = build_user_profile(history, articles_df)
            
            # Normalize user profile
            user_norm = np.linalg.norm(user_profile)
            if user_norm == 0:
                # Skip this user if profile is all zeros
                continue
            user_profile_normalized = user_profile / user_norm
            
            # Use Faiss to quickly find similar articles
            k = articles_df.shape[0]  # Number of articles to retrieve (all)
            similarity_scores = np.zeros(k)
            
            # Search using Faiss
            D, I = index.search(user_profile_normalized.reshape(1, -1).astype(np.float32), k)
            similarity_scores = D[0]  # Similarity scores from Faiss
            
            # Prepare features for XGBoost
            X = similarity_scores.reshape(-1, 1)
            dtest = xgb.DMatrix(X)
            
            # Predict scores
            preds = model.predict(dtest)
            
            # Get top 5 predictions
            top_indices = np.argsort(preds)[-5:][::-1]
            predictions = articles_df.iloc[top_indices]['article_id'].tolist()
            
            # Calculate rank and NDCG
            if ground_truth in predictions:
                rank = predictions.index(ground_truth) + 1
            else:
                rank = None
                
            ndcg = compute_ndcg(rank)
            
            batch_results.append({
                'user_id': uid,
                'ground_truth': ground_truth,
                'predictions': predictions,
                'rank': rank,
                'ndcg': ndcg
            })
            
        all_results.extend(batch_results)
    
    return all_results

# Alternative implementation using NumPy without Faiss
def batch_predict_for_users_numpy(user_groups, articles_df, model, batch_size=32):
    """
    Process predictions for multiple users in batches using pure NumPy
    """
    all_results = []
    
    # Pre-compute article embeddings matrix once (n_articles x embed_dim)
    article_embeddings = articles_df[embed_cols].values
    
    # Process users in batches
    user_ids = list(user_groups.groups.keys())
    
    for i in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_results = []
        
        for uid in batch_user_ids:
            group = user_groups.get_group(uid)
            if len(group) < 2:
                continue  # not enough history
                
            group_sorted = group.sort_values('click_timestamp')
            ground_truth = group_sorted.iloc[-1]['click_article_id']
            
            # Get user profile
            history = group_sorted.iloc[:-1]
            user_profile = build_user_profile(history, articles_df)
            
            # Compute cosine similarity for all articles at once
            similarity_scores = vectorized_cosine_similarity_cpu(user_profile, article_embeddings)
            
            # Prepare features for XGBoost
            X = similarity_scores.reshape(-1, 1)
            dtest = xgb.DMatrix(X)
            
            # Predict scores
            preds = model.predict(dtest)
            
            # Get top 5 predictions
            top_indices = np.argsort(preds)[-5:][::-1]
            predictions = articles_df.iloc[top_indices]['article_id'].tolist()
            
            # Calculate rank and NDCG
            if ground_truth in predictions:
                rank = predictions.index(ground_truth) + 1
            else:
                rank = None
                
            ndcg = compute_ndcg(rank)
            
            batch_results.append({
                'user_id': uid,
                'ground_truth': ground_truth,
                'predictions': predictions,
                'rank': rank,
                'ndcg': ndcg
            })
            
        all_results.extend(batch_results)
    
    return all_results

# Main evaluation function
def evaluate_model(test_clicks, articles_df, model, batch_size=32, use_faiss=True):
    start_time = time.time()
    
    test_groups = test_clicks.groupby('user_id')
    
    try:
        if use_faiss:
            results = batch_predict_for_users(test_groups, articles_df, model, batch_size)
        else:
            results = batch_predict_for_users_numpy(test_groups, articles_df, model, batch_size)
    except Exception as e:
        print(f"Error using Faiss: {e}")
        print("Falling back to NumPy implementation...")
        results = batch_predict_for_users_numpy(test_groups, articles_df, model, batch_size)
    
    # Calculate metrics
    ndcg_scores = [r['ndcg'] for r in results]
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Average NDCG: {avg_ndcg:.4f}")
    print(f"Processed {len(results)} users in {processing_time:.2f} seconds")
    
    # for r in results[:5]:
    #     print(f"User {r['user_id']} - GT: {r['ground_truth']} - Predicted top5: {r['predictions']} - Rank: {r['rank']} - NDCG: {r['ndcg']:.4f}")
    
    return results, avg_ndcg

train_groups = train_clicks.groupby('user_id')
train_data = []
train_labels = []
group_ptr = []

for uid, group in train_groups:
    group = group.sort_values('click_timestamp')
    if len(group) < 2: continue
    
    history = group.iloc[:-1]
    pos_click = group.iloc[-1]
    
    user_profile = build_user_profile(history, articles)
    
    pos_article = articles[articles['article_id'] == pos_click['click_article_id']]
    if pos_article.empty: continue # 会走到这里吗？
    pos_embedding = pos_article[embed_cols].values.flatten()
    
    pos_score = cosine_similarity(user_profile, pos_embedding)
    train_data.append([pos_score])
    train_labels.append(1)
    
    n_negatives = 4
    neg_candidates = articles.sample(n=n_negatives, random_state=42)
    # neg_candidates = articles[articles['article_id'] != pos_click['click_article_id']].sample(n=n_negatives, random_state=42)
    for _, neg in neg_candidates.iterrows():
        neg_embedding = neg[embed_cols].values.flatten()
        neg_score = cosine_similarity(user_profile, neg_embedding)
        train_data.append([neg_score])
        train_labels.append(0)
    
    group_ptr.append(1 + n_negatives)

dtrain = xgb.DMatrix(np.array(train_data), label=np.array(train_labels))
dtrain.set_group(group_ptr)
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
num_round = 50
model = xgb.train(params, dtrain, num_boost_round=num_round)
results, avg_ndcg = evaluate_model(test_clicks, articles, model, batch_size=64)