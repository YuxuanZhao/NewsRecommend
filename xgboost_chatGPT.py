import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import math

articles = pd.read_csv('drive/MyDrive/news/articles.csv')
embed_cols = articles.columns[4:]
train_clicks = pd.read_csv('drive/MyDrive/news/train_click_log.csv')
test_clicks = pd.read_csv('drive/MyDrive/news/test_click_log.csv')
train_clicks = train_clicks.head(1000) # 1112623
test_clicks = test_clicks.head(2) # 518010

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

def predict_for_user(user_clicks, articles_df, model):
    history = user_clicks.sort_values('click_timestamp').iloc[:-1]
    user_profile = build_user_profile(history, articles_df)
    
    # 对所有文章 score，现实中应该是对召回的那一部分
    features = []
    for idx, row in articles_df.iterrows():
        emb = row[embed_cols].values.flatten()
        score = cosine_similarity(user_profile, emb)
        features.append(score)
    
    X = np.array(features).reshape(-1, 1)
    dtest = xgb.DMatrix(X)
    preds = model.predict(dtest)
    
    articles_df['pred_score'] = preds
    top5 = articles_df.sort_values('pred_score', ascending=False).head(5)
    return top5['article_id'].tolist()

test_groups = test_clicks.groupby('user_id')
ndcg_scores = []
results = []

for uid, group in test_groups:
    if len(group) < 2:
        continue  # not enough history
    group_sorted = group.sort_values('click_timestamp')
    ground_truth = group_sorted.iloc[-1]['click_article_id']
    predictions = predict_for_user(group, articles, model)
    
    if ground_truth in predictions:
        rank = predictions.index(ground_truth) + 1  # rank position (1-indexed)
    else:
        rank = None
    
    ndcg = compute_ndcg(rank)
    ndcg_scores.append(ndcg)
    results.append({
        'user_id': uid,
        'ground_truth': ground_truth,
        'predictions': predictions,
        'rank': rank,
        'ndcg': ndcg
    })

# Calculate average NDCG over all users
avg_ndcg = np.mean(ndcg_scores)
print(f"Average NDCG: {avg_ndcg:.4f}")

# Display evaluation results for a few users
for r in results[:5]:
    print(f"User {r['user_id']} - GT: {r['ground_truth']} - Predicted top5: {r['predictions']} - Rank: {r['rank']} - NDCG: {r['ndcg']:.4f}")
