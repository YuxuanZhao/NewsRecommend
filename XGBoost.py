import numpy as np
import xgboost as xgb
import multiprocessing as mp
from sklearn.metrics import ndcg_score
import os

prefix = 'news/'
article_embedding_dict = np.load(prefix + 'article_embedding_dict.npy', allow_pickle=True).item()
train_user_embedding_dict = np.load(prefix + 'train_user_profile.npy', allow_pickle=True).item()
clicked_article_ids = np.load(prefix + 'train_user_clicked_article_ids.npy', allow_pickle=True).item()
test_user_ground_truth = np.load(prefix + 'test_user_ground_truth.npy', allow_pickle=True).item()
test_user_embedding_dict = np.load(prefix + 'test_user_profile.npy', allow_pickle=True).item()
test_user_recommendations = np.load(prefix + 'test_user_recommendations.npy', allow_pickle=True).item()
all_article_ids = list(article_embedding_dict.keys())

if os.path.exists(prefix + "xgboost_model.json"):
    model = xgb.Booster()
    model.load_model(prefix + "xgboost_model.json")
    print('Model loaded')

def preprocess_user(args):
    user_id, article_ids = args
    neg_samples_batch = np.random.choice(all_article_ids, size=3 * len(article_ids), replace=False)
    neg_samples_batch = neg_samples_batch.reshape(len(article_ids), 3)
    user_features = []
    user_labels = []    
    for idx, article_id in enumerate(article_ids):
        user_features.append(np.concatenate([train_user_embedding_dict[user_id], article_embedding_dict[article_id]]))
        user_labels.append(1)
        for neg_art in neg_samples_batch[idx]:
            user_features.append(np.concatenate([train_user_embedding_dict[user_id], article_embedding_dict[int(neg_art)]]))
            user_labels.append(0)
    return user_features, user_labels

def preprocess():
    with mp.get_context('spawn').Pool(mp.cpu_count()) as pool:
        results = pool.map(preprocess_user, list(clicked_article_ids.items()))
    pool.close()
    pool.join()

    X_train = []
    y_train = []
    group = []
    for features, labels in results:
        X_train.extend(features)
        y_train.extend(labels)
        group.append(len(features))
    return X_train, y_train, group

def inference(args):
    user_id, candidates = args
    candidate_features = []
    for art_id in candidates:
        candidate_features.append(np.concatenate([test_user_embedding_dict[user_id], article_embedding_dict[art_id]]))
    
    candidate_features = np.array(candidate_features)
    dtest = xgb.DMatrix(candidate_features)
    scores = model.predict(dtest)
    
    ranked_idx = np.argsort(-scores)[:5]
    ranked_candidates = np.array(candidates)[ranked_idx]
    
    relevance = [1 if int(art) == test_user_ground_truth[user_id] else 0 for art in ranked_candidates]
    return ndcg_score([relevance], [np.arange(5, 0, -1)])

if __name__ == '__main__':
    if not model:
        X_train, y_train, group = preprocess()
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
        model.save_model(prefix + "xgboost_model.json")
        print('Model trained')

    with mp.get_context('spawn').Pool(mp.cpu_count()) as pool:
        results = pool.map(inference, list(test_user_recommendations.items()))
    pool.close()
    pool.join()
    avg_ndcg = np.mean(results)
    print(f'NDCG@5: {avg_ndcg:.4f}')