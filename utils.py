import numpy as np
from collections import defaultdict, Counter

test_click_log = np.load('news/test_click_log.npy')
article_embedding_dict = np.load('news/article_embedding_dict.npy', allow_pickle=True).item()

test_user_ground_truth = {}
embeddings = defaultdict(list)
user_profile = {}
for i, (user_id, art_id) in enumerate(test_click_log):
    if i == len(test_click_log) - 1 or test_click_log[i+1][0] != user_id:
        test_user_ground_truth[int(user_id)] = int(art_id)
        if user_id not in embeddings:
            embeddings[int(user_id)].append(article_embedding_dict[art_id])
    else:
        embeddings[int(user_id)].append(article_embedding_dict[art_id])

for user_id, embs in embeddings.items():
    user_profile[user_id] = np.mean(embs, axis=0)

np.save("news/test_user_ground_truth.npy", test_user_ground_truth)
np.save("news/test_user_profile.npy", user_profile)

# lengths = [len(arr) for arr in embeddings.values()]
# length_distribution = Counter(lengths)
# for length, count in sorted(length_distribution.items()):
#     print(f"Length {length}: {count} keys")