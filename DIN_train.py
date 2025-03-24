import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

prefix = 'news/'
EMBEDDING_DIM = 253
ATTENTION_UNITS = 64
MAX_HISTORY_LENGTH = 50
TRAIN_SET_RATIO = 0.8
BATCH_SIZE = 128
NUM_WORKERS = 20
NUM_EPOCHS = 1

article_embedding_dict = np.load(prefix + 'article_embedding_dict.npy', allow_pickle=True).item()
user_clicked_article_ids = np.load(prefix + 'train_user_clicked_article_ids.npy', allow_pickle=True).item()
user_ids = list(user_clicked_article_ids.keys())
random.shuffle(user_ids)
train_user_ids = user_ids[:int(TRAIN_SET_RATIO * len(user_ids))]
test_user_ids = user_ids[int(TRAIN_SET_RATIO * len(user_ids)):]

class DINDataset(Dataset):
    def __init__(self, user_ids, user_clicked_articles, article_embeddings, max_hist_length=MAX_HISTORY_LENGTH, is_training=True):
        self.user_ids = user_ids
        self.user_clicked_articles = user_clicked_articles
        self.article_embeddings = article_embeddings
        self.max_hist_length = max_hist_length
        self.is_training = is_training
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        article_ids = list(self.article_embeddings.keys())
        
        for user_id in self.user_ids:
            clicked_articles = self.user_clicked_articles[user_id]
            
            if self.is_training: target_indices = list(range(1, len(clicked_articles)))
            else: target_indices = [len(clicked_articles) - 1]
                
            for idx in target_indices:
                target_article = clicked_articles[idx]
                    
                history = clicked_articles[:idx][-self.max_hist_length:]
                
                samples.append({'user_id': user_id, 'history': history, 'target': target_article, 'label': 1})
                
                if self.is_training:
                    negative_article = random.choice(article_ids)
                    while negative_article in clicked_articles:
                        negative_article = random.choice(article_ids)
                                        
                    samples.append({'user_id': user_id, 'history': history, 'target': negative_article, 'label': 0})
        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_emb = self.article_embeddings[sample['target']]
        
        history_emb = np.zeros((self.max_hist_length, EMBEDDING_DIM))
        
        for i, article_id in enumerate(sample['history']):
            if article_id in self.article_embeddings:
                history_emb[i] = self.article_embeddings[article_id]
        
        return {
            'user_id': sample['user_id'],
            'history_emb': torch.FloatTensor(history_emb),
            'target_emb': torch.FloatTensor(target_emb),
            'label': torch.FloatTensor([sample['label']])
        }

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, attention_units):
        super(AttentionLayer, self).__init__()
        self.attention_units = attention_units
        
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, attention_units),
            nn.ReLU(),
            nn.Linear(attention_units, 1),
        )
    
    def forward(self, queries, keys):
        """
        Args:
            queries: target article embedding, shape (batch_size, embedding_dim)
            keys: history article embeddings, shape (batch_size, max_hist_length, embedding_dim)
        """
        batch_size, max_hist_length, embedding_dim = keys.size()
        queries = queries.unsqueeze(1).repeat(1, max_hist_length, 1)
        attention_input = torch.cat([queries, keys], dim=2)
        attention_input = attention_input.view(-1, embedding_dim * 2)
        attention_output = self.attention(attention_input)
        attention_output = attention_output.view(batch_size, max_hist_length)
        attention_weights = F.softmax(attention_output, dim=1)
        output = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        return output, attention_weights

class DIN(nn.Module):
    def __init__(self, embedding_dim, attention_units, fc_units=64):
        super(DIN, self).__init__()
        self.embedding_dim = embedding_dim        
        self.attention = AttentionLayer(embedding_dim, attention_units)        
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, fc_units),
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
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, target_emb, hist_emb):
        """
        Args:
            target_emb: target article embedding, shape (batch_size, embedding_dim)
            hist_emb: history article embeddings, shape (batch_size, max_hist_length, embedding_dim)
            hist_mask: mask for history, shape (batch_size, max_hist_length)
        """
        weighted_hist, attention_weights = self.attention(target_emb, hist_emb)
        concat_feature = torch.cat([target_emb, weighted_hist], dim=1)
        logits = self.fc_layers(concat_feature)
        return logits, attention_weights
    
    def predict(self, target_emb, hist_emb):
        """For inference with sigmoid activation"""
        logits, attention_weights = self.forward(target_emb, hist_emb)
        return self.sigmoid(logits), attention_weights

def train(model, train_loader, optimizer, criterion, device, clip_value=1.0):
    model.train()
    epoch_loss = 0
    processed_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):            
        history_emb, target_emb, labels = batch['history_emb'].to(device), batch['target_emb'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        logits, _ = model(target_emb, history_emb)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        epoch_loss += loss.item()
        processed_batches += 1

    return epoch_loss / max(1, processed_batches)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    processed_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
                
            history_emb = batch['history_emb'].to(device)
            target_emb = batch['target_emb'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(target_emb, history_emb)            
            loss = criterion(logits, labels)            
            probs = torch.sigmoid(logits)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                processed_batches += 1
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics if we have predictions
    accuracy = 0
    if all_preds:
        # Calculate accuracy
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        binary_preds = (all_preds >= 0.5).astype(int)
        accuracy = (binary_preds == all_labels).mean()
    
    # Return average loss over all valid batches
    return total_loss / max(1, processed_batches), accuracy

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = DINDataset(train_user_ids, user_clicked_article_ids, article_embedding_dict, is_training=True)
    test_dataset = DINDataset(test_user_ids, user_clicked_article_ids, article_embedding_dict, is_training=False)
    print(f"Generated {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    
    model = DIN(EMBEDDING_DIM, ATTENTION_UNITS).to(device)    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss() # sigmoid + BCE
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 3
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_din_model.pth')
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping after {epoch+1} epochs without improvement")
            break

    model.load_state_dict(torch.load('best_din_model.pth', weights_only=True))
    final_loss, final_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Final model performance - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")