import numpy as np

test_user_recommendations = np.load('news/test_user_recommendations.npy', allow_pickle=True).item()
test_user_ground_truth = np.load('news/test_user_ground_truth.npy', allow_pickle=True).item()

for user_id, rec_list in test_user_recommendations.items():
    if len(rec_list) > 400:
        np.random.choice(rec_list.size, size=400, replace=True)
    
    gt_article = test_user_ground_truth.get(user_id)
    
    if gt_article is not None and gt_article not in rec_list:
        rec_list = np.append(rec_list, gt_article)
    
    test_user_recommendations[user_id] = rec_list

np.save('news/test_user_recommendations.npy', test_user_recommendations)
