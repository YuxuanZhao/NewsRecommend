import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import optuna

prefix = 'news/'
NUM_WORKERS = 20

# article_embedding_dict
article_emb = np.load(prefix + 'article_dict.npy', allow_pickle=True).item()
article_ids = list(article_emb.keys())
EMBED_DIM = len(article_emb[article_ids[0]])
train_user_clicks = np.load(prefix + 'train_user_clicked_article_ids.npy', allow_pickle=True).item()
test_user_clicks = np.load(prefix + 'test_user_clicked_article_ids.npy', allow_pickle=True).item()
test_user_recs = np.load(prefix + 'test_user_recommendations.npy', allow_pickle=True).item()

class EvalDataset(Dataset):
    def __init__(self, max_history):
        self.max_history = max_history
        self.data = []
        for uid, clicks in test_user_clicks.items():
            if len(clicks) <= 1: continue
            history = clicks[:-1][-max_history:]
            labels = [0 for _ in range(len(test_user_recs[uid]))]
            for i, article_id in enumerate(test_user_recs[uid]):
                if int(article_id) == int(clicks[-1]):
                    labels[i] = 1
                    break
            self.data.append({
                'uid': uid,
                'history': history,
                'candidates': test_user_recs[uid],
                'labels': labels
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        hist_emb = np.zeros((self.max_history, EMBED_DIM))
        for i, aid in enumerate(sample['history']):
            hist_emb[i] = article_emb[aid]
        cand_embs = []
        for aid in sample['candidates']:
            cand_embs.append(article_emb[aid])
        cand_embs = np.array(cand_embs)
        return {
            'uid': sample['uid'],
            'history_emb': torch.FloatTensor(hist_emb),           # (MAX_HIST, EMBED_DIM)
            'cand_embs': torch.FloatTensor(cand_embs),            # (num_candidates, EMBED_DIM)
            'labels': torch.FloatTensor(sample['labels'])         # (num_candidates,)
        }
    
def custom_collate_fn(batch):
    uids = [item['uid'] for item in batch]
    history_emb = torch.stack([item['history_emb'] for item in batch], dim=0)  # (batch, MAX_HIST, EMBED_DIM)
    cand_embs = [item['cand_embs'] for item in batch]  # (batch, num_candidates, EMBED_DIM)
    labels = [item['labels'] for item in batch]  # (batch, num_candidates)
    return {'uid': uids, 'history_emb': history_emb, 'cand_embs': cand_embs, 'labels': labels}

class TrainDataset(Dataset):
    def __init__(self, max_history):
        self.max_history = max_history
        self.samples = []
        for uid, clicks in train_user_clicks.items():
            for i in range(1, len(clicks)):
                history = clicks[:i][-max_history:]
                self.samples.append({'uid': uid, 'history': history, 'target': clicks[i], 'label': 1})
                neg = random.choice(article_ids)
                while neg in clicks: neg = random.choice(article_ids)
                self.samples.append({'uid': uid, 'history': history, 'target': neg, 'label': 0})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_emb = article_emb[sample['target']]
        hist_emb = np.zeros((self.max_history, EMBED_DIM))
        for i, aid in enumerate(sample['history']):
            hist_emb[i] = article_emb[aid]
        return {
            'uid': sample['uid'],
            'history_emb': torch.FloatTensor(hist_emb),
            'target_emb': torch.FloatTensor(target_emb),
            'label': torch.FloatTensor([sample['label']])
        }

class AttentionLayer(nn.Module):
    def __init__(self, emb_dim, attn_units):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(emb_dim * 2, attn_units),
            nn.ReLU(),
            nn.Linear(attn_units, 1),
        )

    def forward(self, query, keys): # (b, d), (b, L, d)
        b, L, d = keys.size()
        query = query.unsqueeze(1).repeat(1, L, 1)
        attn_in = torch.cat([query, keys], dim=2).view(-1, d * 2) # (bL, 2d)
        attn_w = self.attn(attn_in).view(b, L)
        attn_w = F.softmax(attn_w, dim=1)
        # batch matrix multiplication (b, 1, L) (b, L, d) => (b, 1, d) => (b, d)
        out = torch.bmm(attn_w.unsqueeze(1), keys).squeeze(1)
        return out

class DIN(nn.Module):
    def __init__(self, emb_dim, attn_units, fc_units, dropout_rate):
        super().__init__()
        self.attn = AttentionLayer(emb_dim, attn_units)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(emb_dim * 2), 
            nn.Linear(emb_dim * 2, fc_units), nn.ReLU(), nn.Dropout(dropout_rate), nn.BatchNorm1d(fc_units), 
            nn.Linear(fc_units, fc_units // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.BatchNorm1d(fc_units // 2), 
            nn.Linear(fc_units // 2, 1)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, query, history):
        weighted = self.attn(query, history)
        x = torch.cat([query, weighted], dim=1)
        return self.fc(x)

    def predict(self, target, history):
        logits = self(target, history)
        return self.sigmoid(logits)

def train(model, loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train"):
        hist = batch['history_emb'].to(device)
        target = batch['target_emb'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        loss = criterion(model(target, hist), label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, k):
    model.eval()
    total_loss = 0
    ndcgs = []
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            history_batch = batch['history_emb'].to(device)
            batch_uids = batch['uid']
            batch_cand_embs = batch['cand_embs']
            batch_labels = batch['labels']
            
            for i in range(len(batch_uids)):
                his = history_batch[i].unsqueeze(0)  # (1, MAX_HIST, EMBED_DIM)
                cand = batch_cand_embs[i].to(device)   # (num_candidates, EMBED_DIM)
                labels = batch_labels[i].to(device)    # (num_candidates,)
                num_candidates = cand.size(0)
                
                his_expanded = his.expand(num_candidates, -1, -1)  # (num_candidates, MAX_HIST, EMBED_DIM)
                logits = model(cand, his_expanded)  # (num_candidates, 1)
                logits = logits.view(-1)            # (num_candidates,)
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                total_samples += 1

                probs = torch.sigmoid(logits).cpu().numpy()  # (num_candidates,)
                labs = labels.cpu().numpy()
                top_k_idx = np.argsort(-probs)[:k]
                ndcg = 0.0
                for rank, idx in enumerate(top_k_idx, start=1):
                    if labs[idx] == 1:
                        ndcg = 1 / np.log2(rank + 1)
                        break
                ndcgs.append(ndcg)
    
    avg_loss = total_loss / total_samples
    avg_ndcg = np.mean(ndcgs)
    return avg_loss, avg_ndcg

def objective(trial):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    attn_units = trial.suggest_int('attn_units', 32, 128, step=32)
    fc_units = trial.suggest_int('fc_units', 32, 128, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    max_history = trial.suggest_int('max_history', 32, 128, step=32)
    epochs = 2

    train_set = TrainDataset(max_history)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=NUM_WORKERS)
    eval_set = EvalDataset(max_history)
    eval_loader = DataLoader(eval_set, batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate_fn)
    model = DIN(EMBED_DIM, attn_units, fc_units, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        train(model, train_loader, optimizer, criterion, device)
        val_loss, ndcg = evaluate(model, eval_loader, criterion, device, 5)
        scheduler.step(val_loss)
    return ndcg

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 1.62e-3
    weight_decay = 8.96e-5
    attn_units = 128
    fc_units = 32
    dropout_rate = 0.36
    batch_size = 64
    max_history = 64
    epochs = 5

    train_set = TrainDataset(max_history)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=NUM_WORKERS)
    eval_set = EvalDataset(max_history)
    eval_loader = DataLoader(eval_set, batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate_fn)
    model = DIN(EMBED_DIM, attn_units, fc_units, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, ndcg = evaluate(model, eval_loader, criterion, device, 5)
        print(f'Epoch {epoch + 1}: Train {train_loss}, Validate {val_loss}, NDCG {ndcg}')
        scheduler.step(val_loss)

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best:", study.best_trial.params)
    # main()