import numpy as np
import faiss

prefix = 'XGBoost/news/'

articles = np.load(prefix + 'articles.npy')
article_ids = articles[:, 0].astype(np.int64) # (364047,)
embeddings = articles[:, 1:].astype(np.float32) # (364047, 253)
embeddings_size = embeddings.shape[1]

num_clusters = 300
clustering = faiss.Clustering(embeddings_size, num_clusters)
clustering.niter = 80
clustering.verbose = True

index = faiss.IndexHNSWFlat(embeddings_size, 32)
embeddings = np.ascontiguousarray(embeddings)
clustering.train(embeddings, index)
centroids = faiss.vector_float_to_array(clustering.centroids).reshape(num_clusters, embeddings_size)

_, assignments = index.search(embeddings, 1)
assignments = assignments.flatten()
cluster_to_articles = {i: article_ids[assignments == i] for i in range(num_clusters)}

centroid_index = faiss.IndexFlatL2(embeddings_size)
centroid_index.add(centroids)
user_recommendations = {}
user_profiles = np.load(prefix + 'test_user_profile.npy', allow_pickle=True).item()

for uid, profile in user_profiles.items():
    profile = profile.reshape(1, embeddings_size).astype(np.float32)
    _, I = centroid_index.search(profile, 1)
    candidate_article_ids = np.array(cluster_to_articles[int(I[0, 0])])
    user_recommendations[uid] = candidate_article_ids

np.save(prefix + 'user_recommendations.npy', user_recommendations, allow_pickle=True)