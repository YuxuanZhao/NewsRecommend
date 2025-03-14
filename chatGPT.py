import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# 1. Data Processing
# -------------------------

# Load CSV files
train_click_log = pd.read_csv('train_click_log.csv')
articles = pd.read_csv('articles.csv')
articles_emb = pd.read_csv('articles_emb.csv')

# Merge articles with their embeddings (on article_id)
articles_full = pd.merge(articles, articles_emb, on='article_id')

# Create a dictionary of article embeddings (for fast lookup)
emb_cols = [col for col in articles_emb.columns if col.startswith('emb_')]
article2emb = {
    row['article_id']: row[emb_cols].values.astype(np.float32)
    for _, row in articles_emb.iterrows()
}

# Process training user histories:
# For each user, sort clicks by timestamp and get the list of clicked articles.
user_histories = train_click_log.sort_values('click_timestamp') \
                                .groupby('user_id')['click_article_id'] \
                                .apply(list) \
                                .to_dict()

# Generate training samples:
# For each user, use all clicks except the last as history and the last click as the positive target.
# For negative samples, we sample from articles that are not in the user history.
all_article_ids = set(articles['article_id'].unique())
training_samples = []
for user, clicks in user_histories.items():
    if len(clicks) < 2:
        continue
    history = clicks[:-1]
    positive = clicks[-1]
    negatives = list(all_article_ids - set(clicks))
    # Sample a fixed number of negatives (e.g., 4 negatives per positive)
    neg_samples = np.random.choice(negatives, size=4, replace=False)
    training_samples.append({
        'user_id': user,
        'history': history,
        'positive': positive,
        'negatives': neg_samples.tolist()
    })

# Utility: function to compute aggregated user embedding from history
def aggregate_user_embedding(history, article2emb):
    emb_list = [article2emb[aid] for aid in history if aid in article2emb]
    if emb_list:
        return np.mean(emb_list, axis=0)
    else:
        # if no embedding is found (should not happen), return zeros
        return np.zeros(len(next(iter(article2emb.values()))), dtype=np.float32)

# -------------------------
# 2. Candidate Generation
# -------------------------
# In a real system candidate generation might use more sophisticated methods.
# Here we assume that given a user history, we compute an aggregated embedding
# and then select the top candidates by cosine similarity with all article embeddings.

def generate_candidates(user_history, article2emb, top_k=100):
    user_emb = aggregate_user_embedding(user_history, article2emb)
    # Build matrix of article embeddings
    article_ids = list(article2emb.keys())
    emb_matrix = np.array([article2emb[aid] for aid in article_ids])
    # Compute cosine similarity between user embedding and every article embedding
    sim = cosine_similarity(user_emb.reshape(1, -1), emb_matrix).flatten()
    # Get top_k candidates sorted by similarity (highest first)
    top_indices = np.argsort(sim)[::-1][:top_k]
    candidates = [article_ids[i] for i in top_indices]
    return candidates

# -------------------------
# 3. Ranking Model
# -------------------------

# Define a ranking model that takes as input a user embedding and a candidate article embedding
class RankingModel(nn.Module):
    def __init__(self, emb_dim):
        super(RankingModel, self).__init__()
        # input is the concatenated [user_emb, article_emb]
        self.fc1 = nn.Linear(emb_dim * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, user_emb, cand_emb):
        x = torch.cat([user_emb, cand_emb], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        score = self.fc2(x)
        return score

# Hyperparameters
embedding_dim = len(next(iter(article2emb.values())))
model = RankingModel(embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
margin_loss = nn.MarginRankingLoss(margin=1.0)
num_epochs = 5  # adjust as needed

# Training loop using pairwise ranking loss:
# For each training sample, compute the score difference between positive and each negative.
for epoch in range(num_epochs):
    np.random.shuffle(training_samples)
    total_loss = 0.0
    for sample in training_samples:
        # Prepare user aggregated embedding
        user_emb_np = aggregate_user_embedding(sample['history'], article2emb)
        user_emb = torch.tensor(user_emb_np, dtype=torch.float32).unsqueeze(0)  # shape: [1, emb_dim]
        
        # Positive article embedding
        pos_emb_np = article2emb[sample['positive']]
        pos_emb = torch.tensor(pos_emb_np, dtype=torch.float32).unsqueeze(0)  # shape: [1, emb_dim]
        
        # Negative article embeddings (batch of negatives)
        neg_embs = []
        for neg in sample['negatives']:
            neg_embs.append(article2emb[neg])
        neg_embs = torch.tensor(neg_embs, dtype=torch.float32)  # shape: [num_negatives, emb_dim]
        
        # Expand user embedding for negatives
        user_emb_neg = user_emb.expand(neg_embs.shape[0], -1)
        
        # Compute scores
        pos_score = model(user_emb, pos_emb)  # shape: [1, 1]
        neg_scores = model(user_emb_neg, neg_embs)  # shape: [num_negatives, 1]
        
        # Create target for margin ranking loss: we want pos_score > neg_scores by at least margin
        target = torch.ones(neg_scores.shape)
        loss = margin_loss(pos_score.expand_as(neg_scores), neg_scores, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(training_samples):.4f}")

# -------------------------
# Evaluation using Test Click Log
# -------------------------
# For evaluation, we assume that test_click_log.csv contains click sequences for users
# where the last clicked article is the ground truth.
test_click_log = pd.read_csv('test_click_log.csv')
# Get test user histories (each test user appears only in test_click_log)
test_user_histories = test_click_log.sort_values('click_timestamp') \
                                    .groupby('user_id')['click_article_id'] \
                                    .apply(list) \
                                    .to_dict()

def recommend_for_user(user_history, article2emb, model, top_candidates=100, top_final=5):
    # Candidate generation: recall stage
    candidates = generate_candidates(user_history, article2emb, top_k=top_candidates)
    
    # Remove articles already in history if desired
    candidates = [aid for aid in candidates if aid not in user_history]
    
    # Prepare user embedding
    user_emb_np = aggregate_user_embedding(user_history, article2emb)
    user_emb = torch.tensor(user_emb_np, dtype=torch.float32).unsqueeze(0)
    
    scores = []
    for aid in candidates:
        cand_emb_np = article2emb[aid]
        cand_emb = torch.tensor(cand_emb_np, dtype=torch.float32).unsqueeze(0)
        score = model(user_emb, cand_emb)
        scores.append(score.item())
    
    # Rank candidates by score (descending order)
    ranked_candidates = [aid for _, aid in sorted(zip(scores, candidates), reverse=True)]
    return ranked_candidates[:top_final]

# Compute Mean Reciprocal Rank (MRR) over test users:
def compute_mrr(recommended, ground_truth):
    if ground_truth in recommended:
        rank = recommended.index(ground_truth) + 1
        return 1.0 / rank
    else:
        return 0.0

mrr_scores = []
for user, clicks in test_user_histories.items():
    if len(clicks) < 2:
        continue
    history = clicks[:-1]
    ground_truth = clicks[-1]
    recs = recommend_for_user(history, article2emb, model)
    mrr_scores.append(compute_mrr(recs, ground_truth))

print("Mean Reciprocal Rank (MRR) on test data: {:.4f}".format(np.mean(mrr_scores)))
