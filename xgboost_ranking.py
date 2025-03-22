import numpy as np
import xgboost as xgb
import multiprocessing as mp

article_embedding_dict = np.load('news/article_embedding_dict.npy', allow_pickle=True).item()
train_user_embedding_dict = np.load('news/train_user_profile.npy', allow_pickle=True).item()
all_article_ids = list(article_embedding_dict.keys())
clicked_article_ids = np.load('news/train_user_clicked_article_ids.npy', allow_pickle=True).item()

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

if __name__ == '__main__':
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
    model.save_model("xgboost_model.json")