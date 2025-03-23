import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math

prefix = 'news/'
article_embedding_dict = np.load(prefix + "article_embedding_dict.npy", allow_pickle=True).item()
train_user_clicked = np.load(prefix + "train_user_clicked_article_ids.npy", allow_pickle=True).item()
test_user_history = np.load(prefix + "test_user_clicked_article_ids.npy", allow_pickle=True).item()
test_user_ground_truth = np.load(prefix + "test_user_ground_truth.npy", allow_pickle=True).item()
test_user_recommendations = np.load(prefix + "test_user_recommendations.npy", allow_pickle=True).item()
all_article_ids = list(article_embedding_dict.keys())
embedding_dim = 253

class DINTrainDataset(Dataset):
    def __init__(self, user_clicked_dict, article_embedding_dict):
        self.samples = []
        self.article_embedding_dict = article_embedding_dict
        for user_id, clicked_list in user_clicked_dict.items():
            # Ensure user has at least 2 clicks (one for history, one for target)
            if len(clicked_list) < 2:
                continue
            # Use all clicks except last as history and last as positive target.
            history_ids = clicked_list[:-1]
            pos_target = clicked_list[-1]
            # Positive sample
            self.samples.append((history_ids, pos_target, 1))
            # Negative sampling: sample a candidate not in the clicked list.
            neg_candidate = None
            while True:
                neg_candidate = random.choice(all_article_ids)
                if neg_candidate not in clicked_list:
                    break
            self.samples.append((history_ids, neg_candidate, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history_ids, target_id, label = self.samples[idx]
        # Get embeddings for history articles. (list of arrays)
        history_embeds = [self.article_embedding_dict[aid] for aid in history_ids]
        history_embeds = np.array(history_embeds, dtype=np.float32)  # shape: (seq_len, embed_dim)
        # target embedding:
        target_embed = np.array(self.article_embedding_dict[target_id], dtype=np.float32)
        return {
            'history': torch.tensor(history_embeds),   # (seq_len, embed_dim)
            'target': torch.tensor(target_embed),        # (embed_dim,)
            'label': torch.tensor([label], dtype=torch.float32)
        }

def collate_fn(batch):
    # Pad the history sequences so that all have the same length in the batch.
    histories = [sample['history'] for sample in batch]
    targets = torch.stack([sample['target'] for sample in batch])
    labels = torch.stack([sample['label'] for sample in batch])
    seq_lengths = [h.shape[0] for h in histories]
    max_len = max(seq_lengths)
    # Pad histories with zeros.
    padded_histories = []
    for h in histories:
        pad_len = max_len - h.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, h.shape[1])
            h_padded = torch.cat([h, pad], dim=0)
        else:
            h_padded = h
        padded_histories.append(h_padded)
    histories_tensor = torch.stack(padded_histories)
    # Also return the actual lengths for masking if needed.
    return {'history': histories_tensor, 'target': targets, 'label': labels, 'lengths': seq_lengths}

class DIN(nn.Module):
    def __init__(self, embedding_dim, attn_hidden=80, fc_hidden_units=[80, 40]):
        super(DIN, self).__init__()
        # Attention network: takes as input the concatenation of:
        # target, history, (target - history), (target * history)
        self.attention_fc = nn.Sequential(
            nn.Linear(4 * embedding_dim, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1)
        )
        # Fully connected layers for final prediction.
        fc_input_dim = embedding_dim * 2  # aggregated history and target.
        fc_layers = []
        for hidden in fc_hidden_units:
            fc_layers.append(nn.Linear(fc_input_dim, hidden))
            fc_layers.append(nn.ReLU())
            fc_input_dim = hidden
        fc_layers.append(nn.Linear(fc_input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)
        
    def forward(self, target, history, lengths=None):
        """
        target: (batch_size, embed_dim)
        history: (batch_size, seq_len, embed_dim)
        lengths: list of sequence lengths before padding (optional)
        """
        batch_size, seq_len, embed_dim = history.size()
        # Expand target to (batch_size, seq_len, embed_dim)
        target_expanded = target.unsqueeze(1).expand(-1, seq_len, -1)
        # Compute element-wise features for attention.
        attn_input = torch.cat([target_expanded, history, target_expanded - history, target_expanded * history], dim=-1)
        # Compute attention scores.
        attn_scores = self.attention_fc(attn_input).squeeze(-1)  # shape: (batch_size, seq_len)
        # If using padding, mask out padded positions.
        if lengths is not None:
            mask = torch.zeros_like(attn_scores)
            for i, l in enumerate(lengths):
                mask[i, :l] = 1
            # Set scores for padded items to a very small number.
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, seq_len)
        # Weighted sum of history embeddings.
        history_agg = torch.sum(history * attn_weights.unsqueeze(-1), dim=1)  # (batch_size, embed_dim)
        # Concatenate aggregated history and target.
        x = torch.cat([target, history_agg], dim=-1)
        logit = self.fc(x)
        prob = torch.relu(logit)
        return prob

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DIN(embedding_dim).to(device)
model.apply(init_weights)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = DINTrainDataset(train_user_clicked, article_embedding_dict)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

num_epochs = 5  # adjust epochs as needed

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader:
        history = batch['history'].to(device)         # (batch, seq_len, embed_dim)
        target = batch['target'].to(device)             # (batch, embed_dim)
        labels = batch['label'].to(device)              # (batch, 1)
        lengths = batch['lengths']                      # list of original lengths
        
        optimizer.zero_grad()
        preds = model(target, history, lengths)         # (batch, 1)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")