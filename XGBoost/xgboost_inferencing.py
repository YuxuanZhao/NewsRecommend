from sklearn.metrics import ndcg_score
import numpy as np
import xgboost as xgb
import multiprocessing as mp

article_embedding_dict = np.load('news/article_embedding_dict.npy', allow_pickle=True).item()
test_user_embedding_dict = np.load('news/test_user_profile.npy', allow_pickle=True).item()
user_recommendations = np.load('news/user_recommendations.npy', allow_pickle=True).item()
test_user_ground_truth = np.load('news/test_user_ground_truth.npy', allow_pickle=True).item()

model = xgb.Booster()
model.load_model("news/xgboost_model.json")

def inference(args):
    user_id, candidates = args
    candidate_features = []
    for art_id in candidates:
        candidate_features.append(np.concatenate([test_user_embedding_dict[user_id], article_embedding_dict[art_id]]))
    
    candidate_features = np.array(candidate_features)
    dtest = xgb.DMatrix(candidate_features)
    scores = model.predict(dtest)
    
    ranked_idx = np.argsort(-scores)[:50]
    ranked_candidates = np.array(candidates)[ranked_idx]
    
    relevance = [1 if int(art) == test_user_ground_truth[user_id] else 0 for art in ranked_candidates]
    return ndcg_score([relevance], [np.arange(50, 0, -1)])

if __name__ == '__main__':
    with mp.get_context('spawn').Pool(mp.cpu_count()) as pool:
        results = pool.map(inference, list(user_recommendations.items()))
    pool.close()
    pool.join()
    avg_ndcg = np.mean(results)
    print(f'Average NDCG@50: {avg_ndcg:.4f}')