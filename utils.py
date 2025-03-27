import numpy as np
from collections import Counter

# Load the dictionaries from .npy files
ground_truth = np.load('news/test_user_ground_truth.npy', allow_pickle=True).item()
recommendations = np.load('news/test_user_recommendations.npy', allow_pickle=True).item()

# Initialize counters
got_count = 0

# Iterate over each user in the ground truth
for user_id, true_article in ground_truth.items():
    rec_articles = recommendations.get(user_id, [])
    if true_article in rec_articles:
        got_count += 1

print(f"Users that got the ground truth article: {got_count}/50000")

lengths = [len(arr) for arr in recommendations.values()]
length_distribution = Counter(lengths)
for length, count in sorted(length_distribution.items()):
    print(f"Length {length}: {count}")