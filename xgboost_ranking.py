import numpy as np
import xgboost as xgb
from collections import defaultdict
from sklearn.metrics import ndcg_score
import multiprocessing as mp

articles = np.load('news/articles.npy')
article_ids = articles[:, 0].astype(int)
article_embeddings = articles[:, 1:]  # shape: (364047, 253)

article_embedding_dict = {aid: emb for aid, emb in zip(article_ids, article_embeddings)}

train_click_log = np.load('news/train_click_log.npy')

user_embeddings = defaultdict(list)
for user_id, art_id in train_click_log:
    user_embeddings[int(user_id)].append(article_embedding_dict.get(int(art_id)))

user_embedding_dict = {}
for user_id, emb_list in user_embeddings.items():
    emb_list = [emb for emb in emb_list if emb is not None]
    if emb_list:
        user_embedding_dict[user_id] = np.mean(emb_list, axis=0)
    else:
        user_embedding_dict[user_id] = np.zeros(article_embeddings.shape[1])

X_train = []
y_train = []
group = []  # for ranking objective: list of group sizes (per user)

user_pos = defaultdict(set)
for user_id, art_id in train_click_log:
    user_pos[int(user_id)].add(int(art_id))

all_article_ids = list(set(article_ids))

def process_user(args):
    user_id, pos_articles = args
    u_emb = user_embedding_dict.get(user_id)
    if u_emb is None:
        return None, None, 0
    user_features = []
    user_labels = []
    
    pos_articles = list(pos_articles)
    n_pos = len(pos_articles)
    
    # Generate negatives in one batch for this user
    neg_samples_batch = np.random.choice(all_article_ids, size=3 * n_pos, replace=False)
    neg_samples_batch = neg_samples_batch.reshape(n_pos, 3)
    
    for idx, pos_art in enumerate(pos_articles):
        pos_emb = article_embedding_dict.get(pos_art)
        if pos_emb is None:
            continue
        # Positive sample
        feat = np.concatenate([u_emb, pos_emb])
        user_features.append(feat)
        user_labels.append(1)
        # Negative samples for this positive sample
        for neg_art in neg_samples_batch[idx]:
            neg_emb = article_embedding_dict.get(int(neg_art))
            if neg_emb is None:
                continue
            feat_neg = np.concatenate([u_emb, neg_emb])
            user_features.append(feat_neg)
            user_labels.append(0)
    return user_features, user_labels, len(user_features)

# Prepare arguments for each user
args_list = list(user_pos.items())

# Create a pool of processes (using all available CPU cores)
pool = mp.Pool(mp.cpu_count())
results = pool.map(process_user, args_list)
pool.close()
pool.join()

# Collect and combine results
X_train = []
y_train = []
group = []
for features, labels, grp in results:
    if features is not None:
        X_train.extend(features)
        y_train.extend(labels)
        group.append(grp)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group)

params = {
    'objective': 'rank:pairwise',
    'eta': 0.1,
    'gamma': 1.0,
    'min_child_weight': 0.1,
    'max_depth': 6,
    'eval_metric': 'ndcg',
    'seed': 42,
    'verbosity': 1
}

model = xgb.train(params, dtrain, num_boost_round=100)

test_click_log = np.load('news/test_click_log.npy')

user_embeddings = defaultdict(list)
test_user_ground_truth = {}
for user_id, art_id in test_click_log:
    user_embeddings[int(user_id)].append(article_embedding_dict.get(int(art_id)))
    test_user_ground_truth[int(user_id)] = int(art_id)

user_recommendations = np.load('news/user_recommendations.npy', allow_pickle=True).item()

ndcg_scores = []

for user_id, candidates in user_recommendations.items():
    if len(candidates) == 0: continue
    user_id = int(user_id)
    u_emb = user_embedding_dict.get(user_id)
    if u_emb is None:
        u_emb = np.zeros(article_embeddings.shape[1])
    
    candidate_features = []
    for art_id in candidates:
        art_emb = article_embedding_dict.get(int(art_id))
        if art_emb is None:
            art_emb = np.zeros(article_embeddings.shape[1])
        feat = np.concatenate([u_emb, art_emb])
        candidate_features.append(feat)
    
    candidate_features = np.array(candidate_features)
    dtest = xgb.DMatrix(candidate_features)
    scores = model.predict(dtest)
    
    ranked_idx = np.argsort(-scores)
    ranked_candidates = np.array(candidates)[ranked_idx]
    
    ground_truth = test_user_ground_truth.get(user_id)
    relevance = [1 if int(art) == ground_truth else 0 for art in ranked_candidates]
    score = ndcg_score([relevance[:50]], [np.arange(len(relevance[:50]), 0, -1)])
    ndcg_scores.append(score)

avg_ndcg = np.mean(ndcg_scores)
print(f'Average NDCG@50: {avg_ndcg:.4f}')
