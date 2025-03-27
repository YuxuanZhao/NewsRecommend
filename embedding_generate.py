import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

NUM_FEATURE = 253
MARGIN = 1.0
NUM_EPOCHS = 3
BATCH_SIZE = 64
FC_DIM = 512
EMBEDDING_DIM = 256
LR = 1e-3
DROPOUT = 0.13
WEIGHT_DECAY = 5e-5

prefix = 'news/'
articleid_to_feature = np.load(prefix + 'article_embedding_dict.npy', allow_pickle=True).item()
train_user_clicks = np.load(prefix + 'train_user_clicked_article_ids.npy', allow_pickle=True).item()
test_user_clicks = np.load(prefix + 'test_user_clicked_article_ids.npy', allow_pickle=True).item()
all_article_ids = list(articleid_to_feature.keys())

class ArticleTripletDataset(Dataset):
    def __init__(self, isTrain):
        self.triplets = []
        user_clicks = train_user_clicks if isTrain else test_user_clicks
        for _, clicked_articles in user_clicks.items():
            clicked = set(clicked_articles)
            if len(clicked_articles) < 2: continue
            for i in range(len(clicked_articles)-1):
                for j in range(i+1, len(clicked_articles)):
                    anchor = clicked_articles[i]
                    positive = clicked_articles[j]
                    negative = random.choice(all_article_ids)
                    while negative in clicked:
                        negative = random.choice(all_article_ids)
                    self.triplets.append((anchor, positive, negative))
                    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_id, positive_id, negative_id = self.triplets[idx]
        anchor_feat = torch.tensor(articleid_to_feature[anchor_id], dtype=torch.float)
        positive_feat = torch.tensor(articleid_to_feature[positive_id], dtype=torch.float)
        negative_feat = torch.tensor(articleid_to_feature[negative_id], dtype=torch.float)
        return anchor_feat, positive_feat, negative_feat

class ArticleEmbeddingModel(nn.Module):
    def __init__(self, input_dim, fc_dim, embedding_dim, dropout_rate):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(fc_dim),
            nn.Linear(fc_dim, embedding_dim)
        )
        
    def forward(self, x):
        # You can L2-normalize the embedding if you want to use cosine similarity
        embedding = self.fc(x)
        return embedding
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainDataset = ArticleTripletDataset(True)
    testDataset = ArticleTripletDataset(False)
    trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ArticleEmbeddingModel(NUM_FEATURE, fc_dim=FC_DIM, embedding_dim=EMBEDDING_DIM, dropout_rate=DROPOUT).to(device)
    triplet_loss = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in tqdm(trainDataloader, desc='Train'):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            loss = triplet_loss(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * anchor.size(0)
            
        train_loss = running_loss / len(trainDataloader)
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for anchor, positive, negative in tqdm(testDataloader, desc='Eval'):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                emb_anchor = model(anchor)
                emb_positive = model(positive)
                emb_negative = model(negative)
                loss = triplet_loss(emb_anchor, emb_positive, emb_negative)
                running_loss += loss.item() * anchor.size(0)
        eval_loss = running_loss / len(testDataloader)    
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train: {train_loss:.4f}, Test: {train_loss:.4f}")
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), prefix + 'best_eg_model.pth')

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ArticleEmbeddingModel(NUM_FEATURE, fc_dim=FC_DIM, embedding_dim=EMBEDDING_DIM, dropout_rate=DROPOUT).to(device)
    model.load_state_dict(torch.load(prefix + 'best_eg_model.pth', map_location=device, weights_only=True))
    model.eval()
    
    d = {}
    with torch.no_grad():
        for article_id, feature in tqdm(articleid_to_feature.items(), desc="Computing embeddings"):
            feature_tensor = torch.tensor(feature, dtype=torch.float).unsqueeze(0).to(device)
            embedding = model(feature_tensor)
            d[article_id] = embedding.squeeze(0).cpu().numpy()
    np.save('article_dict.npy', d)

    rows = []
    for article_id, values in d.items():
        row = np.append(values, article_id)
        row = np.array(row)
        rows.append(row)

    result_array = np.array(rows, dtype=object)
    np.save("article_table.npy", result_array)

if __name__ == '__main__':
    inference()
