import numpy as np
from DIN_train import DIN
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

prefix = 'news/'
EMBED_DIM = 253
ATTN_UNITS = 64
MAX_HIST = 50
BATCH_SIZE = 128
NUM_WORKERS = 20
EPOCHS = 1

article_emb = np.load(prefix + 'article_embedding_dict.npy', allow_pickle=True).item()
test_user_clicks = np.load(prefix + 'test_user_clicked_article_ids.npy', allow_pickle=True).item()
test_user_recs = np.load(prefix + 'test_user_recommendations.npy', allow_pickle=True).item()

class EvalDataset(Dataset):
    def __init__(self):
        self.data = []
        for uid, clicks in test_user_clicks.items():
            if len(clicks) <= 1: continue
            history = clicks[:-1][-MAX_HIST:]
            candidates = np.append(test_user_recs[uid], [clicks[-1]])
            labels = [0 for _ in range(len(test_user_recs[uid]) + 1)]
            labels[-1] = 1
            self.data.append({
                'uid': uid,
                'history': history,
                'candidates': candidates,
                'labels': labels
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        hist_emb = np.zeros((MAX_HIST, EMBED_DIM))
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
    """
    Custom collate function to handle variable-length candidate embeddings.
    Keeps 'cand_embs' and 'labels' as lists.
    """
    uids = [item['uid'] for item in batch]
    history_emb = torch.stack([item['history_emb'] for item in batch], dim=0)  # shape: (batch, MAX_HIST, EMBED_DIM)
    cand_embs = [item['cand_embs'] for item in batch]  # list of tensors with shape (num_candidates, EMBED_DIM)
    labels = [item['labels'] for item in batch]  # list of tensors with shape (num_candidates,)
    return {'uid': uids, 'history_emb': history_emb, 'cand_embs': cand_embs, 'labels': labels}

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    ndcgs = []
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            # history_emb is a tensor of shape (batch, MAX_HIST, EMBED_DIM)
            history_batch = batch['history_emb'].to(device)
            # cand_embs and labels are lists (one element per sample in the batch)
            batch_uids = batch['uid']
            batch_cand_embs = batch['cand_embs']
            batch_labels = batch['labels']
            
            # Process each sample in the batch individually:
            for i in range(len(batch_uids)):
                his = history_batch[i].unsqueeze(0)  # shape: (1, MAX_HIST, EMBED_DIM)
                cand = batch_cand_embs[i].to(device)   # shape: (num_candidates, EMBED_DIM)
                labels = batch_labels[i].to(device)    # shape: (num_candidates,)
                num_candidates = cand.size(0)
                
                # Expand history to match the number of candidates
                his_expanded = his.expand(num_candidates, -1, -1)  # shape: (num_candidates, MAX_HIST, EMBED_DIM)
                logits = model(cand, his_expanded)  # shape: (num_candidates, 1)
                logits = logits.view(-1)            # shape: (num_candidates,)
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                total_samples += 1

                # Compute NDCG@50 for this sample:
                probs = torch.sigmoid(logits).cpu().numpy()  # (num_candidates,)
                labs = labels.cpu().numpy()
                top_5_idx = np.argsort(-probs)[:5]
                ndcg = 0.0
                for rank, idx in enumerate(top_5_idx, start=1):
                    if labs[idx] == 1:
                        ndcg = 1 / np.log2(rank + 1)
                        break
                ndcgs.append(ndcg)
    
    avg_loss = total_loss / total_samples
    avg_ndcg = np.mean(ndcgs)
    return avg_loss, avg_ndcg


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_set = EvalDataset()
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, collate_fn=custom_collate_fn)
    criterion = nn.BCEWithLogitsLoss()

    model = DIN(EMBED_DIM, ATTN_UNITS).to(device)
    model.load_state_dict(torch.load('best_din_model.pth', map_location=device, weights_only=True))
    _, ndcg = evaluate(model, eval_loader, criterion, device)
    print(f"NDCG: {ndcg:.4f}")
