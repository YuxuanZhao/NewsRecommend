import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

prefix = 'news/'
EMBED_DIM = 253
ATTN_UNITS = 64
MAX_HIST = 50
BATCH_SIZE = 128
NUM_WORKERS = 20
EPOCHS = 1

article_emb = np.load(prefix + 'article_embedding_dict.npy', allow_pickle=True).item()
article_ids = list(article_emb.keys())
train_user_clicks = np.load(prefix + 'train_user_clicked_article_ids.npy', allow_pickle=True).item()

class DINDataset(Dataset):
    def __init__(self):
        self.samples = []
        for uid, clicks in train_user_clicks.items():
            for i in range(1, len(clicks)):
                history = clicks[:i][-MAX_HIST:]
                self.samples.append({'uid': uid, 'history': history, 'target': clicks[i], 'label': 1})
                neg = random.choice(article_ids)
                while neg in clicks: neg = random.choice(article_ids)
                self.samples.append({'uid': uid, 'history': history, 'target': neg, 'label': 0})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_emb = article_emb[sample['target']]
        hist_emb = np.zeros((MAX_HIST, EMBED_DIM))
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
    def __init__(self, emb_dim, attn_units, fc_units=64):
        super().__init__()
        self.attn = AttentionLayer(emb_dim, attn_units)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(emb_dim * 2), 
            nn.Linear(emb_dim * 2, fc_units), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(fc_units), 
            nn.Linear(fc_units, fc_units // 2), nn.ReLU(), nn.Dropout(0.2), nn.BatchNorm1d(fc_units // 2), 
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

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = DINDataset()
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = DIN(EMBED_DIM, ATTN_UNITS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        scheduler.step(train_loss)
        print(f"Epoch {epoch+1}: Train {train_loss:.4f}")
    torch.save(model.state_dict(), 'best_din_model.pth')