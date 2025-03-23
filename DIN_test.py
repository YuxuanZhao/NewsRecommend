# -------------------------------
# Inference and Ranking (NDCG@50)
# -------------------------------
# def ndcg_at_k(ranked_list, ground_truth, k=50):
#     """
#     ranked_list: list of candidate article ids in ranked order
#     ground_truth: the ground truth article id
#     """
#     try:
#         rank = ranked_list.index(ground_truth) + 1  # ranks start at 1
#     except ValueError:
#         return 0.0
#     if rank > k:
#         return 0.0
#     return 1.0 / math.log2(rank + 1)

# model.eval()
# ndcg_scores = []
# with torch.no_grad():
#     for user_id in test_user_recommendations.keys():
#         # Get user history (test set clicked articles)
#         history_ids = test_user_history.get(user_id, [])
#         if len(history_ids) == 0:
#             continue
#         # Convert history to embeddings
#         history_embeds = [article_embedding_dict[aid] for aid in history_ids if aid in article_embedding_dict]
#         if len(history_embeds) == 0:
#             continue
#         history_tensor = torch.tensor(np.array(history_embeds, dtype=np.float32)).unsqueeze(0).to(device)  # (1, seq_len, embed_dim)
#         candidate_ids = test_user_recommendations[user_id]
#         candidate_probs = []
#         # For each candidate article, get its embedding and compute click probability.
#         for cid in candidate_ids:
#             if cid not in article_embedding_dict:
#                 candidate_probs.append(0.0)
#                 continue
#             target_embed = torch.tensor(article_embedding_dict[cid], dtype=torch.float32).unsqueeze(0).to(device)  # (1, embed_dim)
#             prob = model(target_embed, history_tensor, lengths=[history_tensor.shape[1]])
#             candidate_probs.append(prob.item())
#         # Rank candidates by probability in descending order.
#         ranked_candidates = [cid for _, cid in sorted(zip(candidate_probs, candidate_ids), reverse=True)]
#         gt_article = test_user_ground_truth.get(user_id, None)
#         if gt_article is not None:
#             ndcg = ndcg_at_k(ranked_candidates, gt_article, k=50)
#             ndcg_scores.append(ndcg)

# avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
# print("Average NDCG@50:", avg_ndcg)
