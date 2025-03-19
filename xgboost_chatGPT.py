import pandas as pd
import numpy as np
import xgboost as xgb
import math
import faiss
import time

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0: return 0
    return np.dot(u, v) / (norm_u * norm_v)

def build_user_profile(clicks, articles_df):
    merged = clicks.merge(articles_df, left_on='click_article_id', right_on='article_id', how='left')
    profile = merged[embed_cols].mean().values
    return profile

def compute_ndcg(rank):
    if rank is None: return 0.0
    return 1.0 / math.log2(rank + 1)

def vectorized_cosine_similarity_cpu(user_profile, article_embeddings):
    user_norm = np.linalg.norm(user_profile)
    if user_norm == 0: return np.zeros(article_embeddings.shape[0])
    user_profile_normalized = user_profile / user_norm
    
    article_norms = np.linalg.norm(article_embeddings, axis=1)
    valid_indices = article_norms != 0
    # scores = np.zeros(article_embeddings.shape[0])
    article_embeddings_normalized = np.zeros_like(article_embeddings)
    article_embeddings_normalized[valid_indices] = article_embeddings[valid_indices] / article_norms[valid_indices].reshape(-1, 1)
    
    scores = np.dot(article_embeddings_normalized, user_profile_normalized)
    
    return scores

def batch_predict_for_users(user_groups, articles_df, model, batch_size, topk):
    all_results = []    
    article_embeddings = articles_df[embed_cols].values
    
    article_norms = np.linalg.norm(article_embeddings, axis=1)
    valid_indices = article_norms != 0
    article_embeddings_normalized = np.zeros_like(article_embeddings)
    article_embeddings_normalized[valid_indices] = article_embeddings[valid_indices] / article_norms[valid_indices].reshape(-1, 1)
    
    d = article_embeddings.shape[1]  # dimension
    index = faiss.IndexFlatIP(d)  # Inner product is equivalent to cosine similarity for normalized vectors
    index.add(article_embeddings_normalized.astype(np.float32))  # Add normalized vectors
    
    user_ids = list(user_groups.groups.keys())
    
    for i in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_results = []
        
        for uid in batch_user_ids:
            group = user_groups.get_group(uid)
            if len(group) < 2:
                continue
                
            group_sorted = group.sort_values('click_timestamp')
            ground_truth = group_sorted.iloc[-1]['click_article_id']
            
            history = group_sorted.iloc[:-1]
            user_profile = build_user_profile(history, articles_df)
            
            user_norm = np.linalg.norm(user_profile)
            if user_norm == 0:
                continue
            user_profile_normalized = user_profile / user_norm
            
            k = articles_df.shape[0]
            similarity_scores = np.zeros(k)
            
            D, I = index.search(user_profile_normalized.reshape(1, -1).astype(np.float32), k)
            similarity_scores = D[0]
            
            X = similarity_scores.reshape(-1, 1)
            dtest = xgb.DMatrix(X)
            
            preds = model.predict(dtest)
            
            top_indices = np.argsort(preds)[-topk:][::-1]
            predictions = articles_df.iloc[top_indices]['article_id'].tolist()
            
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

def evaluate_model(test_clicks, articles_df, model, batch_size):
    
    test_groups = test_clicks.groupby('user_id')
    results = batch_predict_for_users(test_groups, articles_df, model, batch_size)
    ndcg_scores = [r['ndcg'] for r in results]
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        
    print(f"Average NDCG: {avg_ndcg:.4f}")

    return results, avg_ndcg

start_time = time.time()

articles = pd.read_csv('drive/MyDrive/news/articles.csv')
embed_cols = articles.columns[4:]
train_clicks = pd.read_csv('drive/MyDrive/news/train_click_log.csv')
test_clicks = pd.read_csv('drive/MyDrive/news/test_click_log.csv')
# train_clicks = train_clicks.head(100000) # 1112623
# test_clicks = test_clicks.head(100) # 518010

print(f"Read csv: {time.time() - start_time:.6f} seconds")
start_time = time.time()

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
    if pos_article.empty: continue
    pos_embedding = pos_article[embed_cols].values.flatten()
    
    pos_score = cosine_similarity(user_profile, pos_embedding)
    train_data.append([pos_score])
    train_labels.append(1)
    
    n_negatives = 4
    neg_candidates = articles.sample(n=n_negatives, random_state=42)
    for _, neg in neg_candidates.iterrows():
        neg_embedding = neg[embed_cols].values.flatten()
        neg_score = cosine_similarity(user_profile, neg_embedding)
        train_data.append([neg_score])
        train_labels.append(0)
    
    group_ptr.append(1 + n_negatives)

print(f"Prepare data: {time.time() - start_time:.6f} seconds")
start_time = time.time()

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

print(f"Train model: {time.time() - start_time:.6f} seconds")
start_time = time.time()

test_groups = test_clicks.groupby('user_id')
results = batch_predict_for_users(test_groups, articles, model, 64, 100)
ndcg_scores = [r['ndcg'] for r in results]
avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0

print(f"Validate model: {time.time() - start_time:.6f} seconds")    
print(f"Average NDCG: {avg_ndcg:.4f}")