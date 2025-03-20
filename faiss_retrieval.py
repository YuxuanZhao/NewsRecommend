import pandas as pd
import numpy as np
import faiss
import time
from tqdm import tqdm

model_start = time.time()
articles_df = pd.read_csv('news/articles.csv')
article_ids = articles_df['article_id'].values
embeddings = articles_df.iloc[:, 4:].values.astype('float32')
article_id_to_index = {aid: idx for idx, aid in enumerate(article_ids)}

dim = embeddings.shape[1]  # should be 250
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"Faiss index built with {index.ntotal} articles in {time.time() - model_start:.2f} seconds")
model_start = time.time()
dtypes = {
    'user_id': 'int32',
    'click_article_id': 'int32',
    'click_timestamp': 'int64',
    'click_environment': 'int8',
    'click_deviceGroup': 'int8',
    'click_os': 'int8',
    'click_country': 'int8',
    'click_region': 'int8',
    'click_referrer_type': 'int8'
}
col_names = [
    'user_id', 'click_article_id', 'click_timestamp',
    'click_environment', 'click_deviceGroup', 'click_os',
    'click_country', 'click_region', 'click_referrer_type'
]
clicks_df = pd.read_csv('news/test_click_log.csv', header=None, names=col_names, dtype=dtypes)

# Group clicks by user_id and compute the mean embedding per user.
user_profiles = {}
for user, group in clicks_df.groupby('user_id'):
    clicked_article_ids = group['click_article_id'].values
    vectors = []
    for aid in clicked_article_ids:
        if aid in article_id_to_index:
            vectors.append(embeddings[article_id_to_index[aid]])
        else:
            continue
    if len(vectors) == 0:
        user_profiles[user] = np.zeros(dim, dtype='float32')
    else:
        user_profiles[user] = np.mean(np.stack(vectors), axis=0)

print(f"Computed profiles for {len(user_profiles)} users from test_click_log in {time.time() - model_start:.2f} seconds")
model_start = time.time()

# ---------- Step 4: Retrieve 1000 nearest neighbor articles for each user ----------
results = []
for user, profile in tqdm(user_profiles.items()):
    # Faiss expects a 2D array for queries.
    profile_vector = np.expand_dims(profile, axis=0)
    distances, indices = index.search(profile_vector, 1000)
    
    # Retrieve the article IDs corresponding to the indices.
    neighbor_article_ids = article_ids[indices[0]]
    results.append({
        'user_id': user,
        'neighbor_article_ids': ','.join(map(str, neighbor_article_ids))
    })

print(f"Retrieved nearest neighbor articles for all users in {time.time() - model_start:.2f} seconds")
model_start = time.time()

# ---------- Step 5: Save the recommendations locally ----------
results_df = pd.DataFrame(results)
results_df.to_csv('user_nn_recommendations.csv', index=False)
print(f"Recommendations saved to user_nn_recommendations.csv in {time.time() - model_start:.2f} seconds")