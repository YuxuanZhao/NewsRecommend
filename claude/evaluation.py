import numpy as np
import pandas as pd

def mean_reciprocal_rank(predictions, ground_truth):
    """
    Calculate the Mean Reciprocal Rank (MRR)
    
    Args:
        predictions: Dictionary of user_id -> list of predicted article_ids
        ground_truth: Dictionary of user_id -> ground truth article_id
    
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    
    for user_id, gt_article_id in ground_truth.items():
        if user_id not in predictions:
            # If no prediction for this user, rank is 0
            reciprocal_ranks.append(0)
            continue
            
        pred_articles = predictions[user_id]
        
        # Find position of ground truth article in predictions
        if gt_article_id in pred_articles:
            rank = pred_articles.index(gt_article_id) + 1  # +1 because index is 0-based
            reciprocal_rank = 1.0 / rank
        else:
            reciprocal_rank = 0
            
        reciprocal_ranks.append(reciprocal_rank)
    
    # Calculate mean
    mrr = np.mean(reciprocal_ranks)
    return mrr

# Convert test_users_last_clicks to a dictionary
ground_truth = dict(zip(test_users_last_clicks['user_id'], test_users_last_clicks['click_article_id']))

# Calculate MRR
mrr_score = mean_reciprocal_rank(test_predictions, ground_truth)

print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")

# Analyze results
def analyze_results(predictions, ground_truth):
    """Generate a detailed analysis of the results"""
    # Calculate hit rate at different positions
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    
    for user_id, gt_article_id in ground_truth.items():
        if user_id not in predictions:
            continue
            
        pred_articles = predictions[user_id]
        
        if gt_article_id in pred_articles[:1]:
            hit_at_1 += 1
            
        if gt_article_id in pred_articles[:3]:
            hit_at_3 += 1
            
        if gt_article_id in pred_articles[:5]:
            hit_at_5 += 1
    
    # Calculate metrics
    total_users = len(ground_truth)
    hit_rate_1 = hit_at_1 / total_users
    hit_rate_3 = hit_at_3 / total_users
    hit_rate_5 = hit_at_5 / total_users
    
    print(f"Total test users: {total_users}")
    print(f"Hit rate @1: {hit_rate_1:.4f}")
    print(f"Hit rate @3: {hit_rate_3:.4f}")
    print(f"Hit rate @5: {hit_rate_5:.4f}")
    
    return {
        'total_users': total_users,
        'hit_rate_1': hit_rate_1,
        'hit_rate_3': hit_rate_3,
        'hit_rate_5': hit_rate_5,
        'mrr': mrr_score
    }

# Analyze results
analysis = analyze_results(test_predictions, ground_truth)

# Save results
results_df = pd.DataFrame.from_dict(analysis, orient='index').T
results_df.to_csv('model_results.csv', index=False)
