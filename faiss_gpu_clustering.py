import numpy as np
import faiss
import torch

articles = np.load('news/articles.npy')
article_ids = articles[:, 0].astype(np.int64)
embeddings = articles[:, 1:].astype(np.float32)
d = embeddings.shape[1]

n_clusters = 350

clustering = faiss.Clustering(d, n_clusters)
clustering.niter = 20
clustering.verbose = True

index = faiss.IndexFlatL2(d)

embeddings = np.ascontiguousarray(embeddings)
clustering.train(embeddings, index)

centroids = faiss.vector_float_to_array(clustering.centroids).reshape(n_clusters, d)


D, assignments = index.search(embeddings, 1)
assignments = assignments.flatten()


cluster_to_articles = {i: article_ids[assignments == i] for i in range(n_clusters)}

article_id_to_embedding = {aid: emb for aid, emb in zip(article_ids, embeddings)}

click_log = np.load('news/test_click_log.npy')
user_ids = click_log[:, 0].astype(np.int64)
click_article_ids = click_log[:, 1].astype(np.int64)

user_to_embeddings = {}
for uid, aid in zip(user_ids, click_article_ids):
    if aid in article_id_to_embedding:
        user_to_embeddings.setdefault(uid, []).append(article_id_to_embedding[aid])

user_profiles = {}
for uid, emb_list in user_to_embeddings.items():
    user_profiles[uid] = np.mean(emb_list, axis=0)

centroid_index = faiss.IndexFlatL2(d)
centroid_index.add(centroids)


user_to_result_articles = {}
for uid, profile in user_profiles.items():
    profile = profile.reshape(1, d).astype(np.float32)
    _, I = centroid_index.search(profile, 1)
    nearest_cluster = int(I[0, 0])
    candidate_article_ids = np.array(cluster_to_articles[nearest_cluster])
    user_to_result_articles[uid] = candidate_article_ids

np.save('news/user_recommendations.npy', user_to_result_articles, allow_pickle=True)