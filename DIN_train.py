import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

# Configurations
prefix = 'news/'
EMBED_DIM = 253
ATTN_UNITS = 64
MAX_HIST = 50
TRAIN_RATIO = 0.8
BATCH_SIZE = 128
NUM_WORKERS = 20
EPOCHS = 1

# Load data
article_emb = np.load(prefix + 'article_embedding_dict.npy', allow_pickle=True).item()
user_clicks = np.load(prefix + 'train_user_clicked_article_ids.npy', allow_pickle=True).item()
user_ids = list(user_clicks.keys())
random.shuffle(user_ids)
split = int(TRAIN_RATIO * len(user_ids))
train_ids, test_ids = user_ids[:split], user_ids[split:]

class DINDataset(Dataset):
    def __init__(self, user_ids, user_clicks, article_emb, max_hist=MAX_HIST, is_train=True):
        self.samples = []
        articles = list(article_emb.keys())
        for uid in user_ids:
            clicks = user_clicks[uid]
            targets = range(1, len(clicks)) if is_train else [len(clicks)-1]
            for i in targets:
                history = clicks[:i][-max_hist:]
                self.samples.append({'uid': uid, 'history': history, 'target': clicks[i], 'label': 1})
                if is_train:
                    neg = random.choice(articles)
                    while neg in clicks:
                        neg = random.choice(articles)
                    self.samples.append({'uid': uid, 'history': history, 'target': neg, 'label': 0})
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_emb = article_emb[sample['target']]
        hist_emb = np.zeros((MAX_HIST, EMBED_DIM))
        for i, aid in enumerate(sample['history']):
            if aid in article_emb:
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

    def forward(self, query, keys):
        b, L, d = keys.size()
        query = query.unsqueeze(1).repeat(1, L, 1)
        attn_in = torch.cat([query, keys], dim=2).view(-1, d * 2)
        attn_w = self.attn(attn_in).view(b, L)
        attn_w = F.softmax(attn_w, dim=1)
        out = torch.bmm(attn_w.unsqueeze(1), keys).squeeze(1)
        return out, attn_w

class DIN(nn.Module):
    def __init__(self, emb_dim, attn_units, fc_units=64):
        super().__init__()
        self.attn = AttentionLayer(emb_dim, attn_units)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(emb_dim * 2),
            nn.Linear(emb_dim * 2, fc_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(fc_units),
            nn.Linear(fc_units, fc_units // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(fc_units // 2),
            nn.Linear(fc_units // 2, 1),
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target, history):
        weighted, attn_w = self.attn(target, history)
        x = torch.cat([target, weighted], dim=1)
        return self.fc(x), attn_w

    def predict(self, target, history):
        logits, attn_w = self(target, history)
        return self.sigmoid(logits), attn_w

def train(model, loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss, count = 0, 0
    for batch in tqdm(loader, desc="Train"):
        hist = batch['history_emb'].to(device)
        target = batch['target_emb'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()
        loss = criterion(model(target, hist)[0], label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, count = 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            hist = batch['history_emb'].to(device)
            target = batch['target_emb'].to(device)
            label = batch['label'].to(device)
            logits, _ = model(target, hist)
            loss = criterion(logits, label)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                count += 1
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                labels.extend(label.cpu().numpy())
    acc = np.mean((np.array(preds).flatten() >= 0.5).astype(int) == np.array(labels).flatten()) if preds else 0
    return total_loss / max(count, 1), acc

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = DINDataset(train_ids, user_clicks, article_emb, is_train=True)
    test_set = DINDataset(test_ids, user_clicks, article_emb, is_train=False)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = DIN(EMBED_DIM, ATTN_UNITS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()

    best_loss, patience = float('inf'), 0
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}, Acc {val_acc:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_din_model.pth')
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                print("Early stopping")
                break

    model.load_state_dict(torch.load('best_din_model.pth', map_location=device))
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final: Loss {final_loss:.4f}, Acc {final_acc:.4f}")
