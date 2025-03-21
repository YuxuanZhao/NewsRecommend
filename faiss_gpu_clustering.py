import numpy as np
import faiss
import torch
import time

start = time.time()

articles = np.load('news/articles.npy')
article_ids = articles[:, 0].astype(np.int64)         # shape: (364047,)
embeddings = articles[:, 1:].astype(np.float32)         # shape: (364047, 253)
d = embeddings.shape[1]  # dimensionality = 253

n_clusters = 350

clustering = faiss.Clustering(d, n_clusters)
clustering.niter = 20         # iterations for clustering
clustering.verbose = True

index = faiss.IndexFlatL2(d) # L2 distance

embeddings = np.ascontiguousarray(embeddings)
clustering.train(embeddings, index)

centroids = faiss.vector_float_to_array(clustering.centroids).reshape(n_clusters, d)

# # Now, assign each article to the nearest centroid.
# # We use the index (which contains the centroids) to search each embedding.
D, assignments = index.search(embeddings, 1)
assignments = assignments.flatten()  # shape: (364047,)

# # Build a mapping from cluster index to all article_ids in that cluster.
cluster_to_articles = {i: article_ids[assignments == i] for i in range(n_clusters)}

# # -------------------------------
# # 3. Prepare article embedding lookup for user profile computation
# # -------------------------------
# # We create a dictionary mapping article_id to its embedding.
# # (Assumes article_ids are unique.)
article_id_to_embedding = {aid: emb for aid, emb in zip(article_ids, embeddings)}

# # -------------------------------
# # 4. Load click log and compute per-user average embeddings
# # -------------------------------
# # test_click_log.npy: first column is user_id and second column is click_article_id.
click_log = np.load('news/test_click_log.npy')
user_ids = click_log[:, 0].astype(np.int64)
click_article_ids = click_log[:, 1].astype(np.int64)

# # Aggregate embeddings for each user
user_to_embeddings = {}
for uid, aid in zip(user_ids, click_article_ids):
    # Only process if the article exists in our lookup.
    if aid in article_id_to_embedding:
        user_to_embeddings.setdefault(uid, []).append(article_id_to_embedding[aid])

# # Compute the average (profile) for each user.
user_profiles = {}
for uid, emb_list in user_to_embeddings.items():
    # Average along axis 0 gives a (253,)-dim vector.
    user_profiles[uid] = np.mean(emb_list, axis=0)

# # -------------------------------
# # 5. For each user, determine nearest centroid and select candidate articles
# # -------------------------------
# # Create a FAISS index for the centroids for fast nearest neighbor search.
centroid_index = faiss.IndexFlatL2(d)
centroid_index.add(centroids)  # centroids shape: (350, 253)

# # For each user, find the nearest centroid and get all article_ids in that cluster.
user_to_result_articles = {}
for uid, profile in user_profiles.items():
    profile = profile.reshape(1, d).astype(np.float32)  # shape: (1, 253)
    _, I = centroid_index.search(profile, 1)             # I: nearest centroid index
    nearest_cluster = int(I[0, 0])
    # Get all articles in the nearest cluster (cell)
    candidate_article_ids = cluster_to_articles[nearest_cluster]
    # Save as a PyTorch tensor (of type int64)
    user_to_result_articles[uid] = torch.tensor(candidate_article_ids, dtype=torch.int64)

# # -------------------------------
# # 6. Save the result locally as a PyTorch file
# # -------------------------------
# # The result is a dictionary mapping each user_id to a tensor of article_ids.
torch.save(user_to_result_articles, 'user_recommendations.pt')

print("User recommendations saved to 'user_recommendations.pt'")
print(time.time() - start)
