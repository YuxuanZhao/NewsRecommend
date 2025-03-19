import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

articles = pd.read_csv('drive/MyDrive/news/articles.csv')
click_log = pd.read_csv('drive/MyDrive/news/clicks.csv')
click_log = click_log.head(1000) # 1000/1630633

pos_data = pd.merge(click_log, articles, left_on='click_article_id', right_on='article_id', how='inner')
pos_data['label'] = 1  # 正样本

all_article_ids = set(articles['article_id'].values)
articles_indexed = articles.set_index('article_id')

neg_ratio = 4
train_data = []
group_sizes = []  # 每个用户有几个样本

for user, group in pos_data.groupby('user_id'):
    clicked_articles = set(group['click_article_id'].unique())
    
    neg_article_ids = list(all_article_ids - clicked_articles)
    n_neg = min(len(neg_article_ids), neg_ratio * len(group))
    neg_sample_ids = np.random.choice(neg_article_ids, size=n_neg, replace=False)
    neg_samples = articles_indexed.loc[neg_sample_ids].reset_index()
    neg_samples['label'] = 0  # 负样本
    neg_samples['user_id'] = user
    
    user_group = pd.concat([group, neg_samples], ignore_index=True)
    train_data.append(user_group)
    group_sizes.append(user_group.shape[0])

train_df = pd.concat(train_data, ignore_index=True)

exclude_columns = ['click_article_id', 'click_timestamp', 
                   'click_environment', 'click_deviceGroup', 'click_os', 
                   'click_country', 'click_region', 'click_referrer_type', 
                   'article_id', 'label']
feature_columns = [col for col in train_df.columns if col not in exclude_columns]

X = train_df[feature_columns].values
y = train_df['label'].values

dtrain = xgb.DMatrix(X, label=y)
dtrain.set_group(group_sizes)

params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'eta': 0.1,
    'max_depth': 6,
    'seed': 42
}

num_round = 50
model = xgb.train(params, dtrain, num_round)

plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=20)
plt.tight_layout()
plt.savefig('feature_importance.png')

model.save_model('article_ranking_model.bin')

def rank_articles(user_id, candidate_article_ids):
    candidates = articles_indexed.loc[candidate_article_ids].reset_index()
    candidates['user_id'] = user_id
    X_candidates = candidates.drop(columns=['article_id']).values
    
    dtest = xgb.DMatrix(X_candidates)
    preds = model.predict(dtest)
    candidates['score'] = preds
    
    ranked = candidates.sort_values('score', ascending=False)
    return ranked['article_id'].tolist()

user_id_example = 123
candidate_article_ids_example = list(np.random.choice(list(all_article_ids), size=10, replace=False))
print("Ranked Articles:", rank_articles(user_id_example, candidate_article_ids_example))