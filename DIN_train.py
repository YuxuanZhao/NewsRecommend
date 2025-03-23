import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

# Load the data
article_embedding_dict = np.load('article_embedding_dict.npy', allow_pickle=True).item()
user_clicked_article_ids = np.load('train_user_clicked_article_ids.npy', allow_pickle=True).item()

# Define constants
EMBEDDING_DIM = 253  # The dimension of article embeddings
ATTENTION_UNITS = 64  # Number of units in attention layer
MAX_HIST_LENGTH = 50  # Maximum length of click history to consider

# Create train/test split
user_ids = list(user_clicked_article_ids.keys())
random.shuffle(user_ids)
train_user_ids = user_ids[:int(0.8 * len(user_ids))]
test_user_ids = user_ids[int(0.8 * len(user_ids)):]

class DINDataset(Dataset):
    def __init__(self, user_ids, user_clicked_articles, article_embeddings, max_hist_length=MAX_HIST_LENGTH, is_training=True):
        self.user_ids = user_ids
        self.user_clicked_articles = user_clicked_articles
        self.article_embeddings = article_embeddings
        self.max_hist_length = max_hist_length
        self.is_training = is_training
        self.samples = self._generate_samples()

    def _generate_samples(self):
        """Generate positive and negative samples for training"""
        samples = []
        article_ids = list(self.article_embeddings.keys())
        
        for user_id in self.user_ids:
            clicked_articles = self.user_clicked_articles[user_id]
            if len(clicked_articles) < 2:
                continue  # Skip users with too few clicks
            
            # Use the last click as target for testing, else use random clicks as targets
            if self.is_training:
                target_indices = list(range(1, len(clicked_articles)))
            else:
                target_indices = [len(clicked_articles) - 1]
                
            for idx in target_indices:
                target_article = clicked_articles[idx]
                
                # Skip if target article is not in embedding dictionary
                if target_article not in self.article_embeddings:
                    continue
                    
                history = clicked_articles[:idx][-self.max_hist_length:]  # Get history before target
                
                # Skip if history is empty
                if len(history) == 0:
                    continue
                    
                # Check if at least one history article is in embedding dictionary
                valid_history = False
                for article_id in history:
                    if article_id in self.article_embeddings:
                        valid_history = True
                        break
                
                if not valid_history:
                    continue
                
                # Positive sample
                samples.append({
                    'user_id': user_id,
                    'history': history,
                    'target': target_article,
                    'label': 1
                })
                
                # Negative sample - randomly sample article not in user's click history
                if self.is_training:
                    negative_article = random.choice(article_ids)
                    retry_count = 0
                    while negative_article in clicked_articles and retry_count < 10:
                        negative_article = random.choice(article_ids)
                        retry_count += 1
                    
                    # Skip if we couldn't find a good negative sample
                    if negative_article in clicked_articles:
                        continue
                    
                    samples.append({
                        'user_id': user_id,
                        'history': history,
                        'target': negative_article,
                        'label': 0
                    })
        
        # Shuffle samples
        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get target article embedding
        target_emb = self.article_embeddings[sample['target']]
        
        # Get history article embeddings
        history = sample['history']
        history_length = len(history)
        history_emb = np.zeros((self.max_hist_length, EMBEDDING_DIM))
        
        valid_count = 0
        for i, article_id in enumerate(history):
            if article_id in self.article_embeddings:
                if valid_count < self.max_hist_length:
                    history_emb[valid_count] = self.article_embeddings[article_id]
                    valid_count += 1
        
        # Create mask for valid history items
        history_mask = np.zeros(self.max_hist_length)
        history_mask[:valid_count] = 1
        
        return {
            'user_id': sample['user_id'],
            'history_emb': torch.FloatTensor(history_emb),
            'history_mask': torch.FloatTensor(history_mask),
            'target_emb': torch.FloatTensor(target_emb),
            'label': torch.FloatTensor([sample['label']])
        }

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, attention_units):
        super(AttentionLayer, self).__init__()
        self.attention_units = attention_units
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, attention_units),
            nn.ReLU(),
            nn.Linear(attention_units, 1),
        )
    
    def forward(self, queries, keys, keys_mask):
        """
        Args:
            queries: target article embedding, shape (batch_size, embedding_dim)
            keys: history article embeddings, shape (batch_size, max_hist_length, embedding_dim)
            keys_mask: mask for history, shape (batch_size, max_hist_length)
        """
        batch_size, max_hist_length, embedding_dim = keys.size()
        
        # Repeat queries to match the history dimension
        queries = queries.unsqueeze(1).repeat(1, max_hist_length, 1)
        
        # Concatenate queries and keys
        attention_input = torch.cat([queries, keys], dim=2)
        
        # Reshape for attention network
        attention_input = attention_input.view(-1, embedding_dim * 2)
        
        # Calculate attention scores
        attention_output = self.attention(attention_input)
        attention_output = attention_output.view(batch_size, max_hist_length)
        
        # Apply mask - set masked positions to negative infinity for softmax
        paddings = torch.ones_like(attention_output) * (-2**32 + 1)
        attention_scores = torch.where(keys_mask.eq(1), attention_output, paddings)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply weights to keys
        output = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        
        return output, attention_weights

class DIN(nn.Module):
    def __init__(self, embedding_dim, attention_units, fc_units=64):
        super(DIN, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Attention layer
        self.attention = AttentionLayer(embedding_dim, attention_units)
        
        # Fully connected layers for final prediction
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
        
        # Separate sigmoid for better numerical stability
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, target_emb, hist_emb, hist_mask):
        """
        Args:
            target_emb: target article embedding, shape (batch_size, embedding_dim)
            hist_emb: history article embeddings, shape (batch_size, max_hist_length, embedding_dim)
            hist_mask: mask for history, shape (batch_size, max_hist_length)
        """
        # Apply attention mechanism
        weighted_hist, attention_weights = self.attention(target_emb, hist_emb, hist_mask)
        
        # Concatenate target embedding with attention-pooled history
        concat_feature = torch.cat([target_emb, weighted_hist], dim=1)
        
        # Final prediction without sigmoid (will use BCEWithLogitsLoss)
        logits = self.fc_layers(concat_feature)
        
        return logits, attention_weights
    
    def predict(self, target_emb, hist_emb, hist_mask):
        """For inference with sigmoid activation"""
        logits, attention_weights = self.forward(target_emb, hist_emb, hist_mask)
        probs = self.sigmoid(logits)
        return probs, attention_weights

def train(model, train_loader, optimizer, criterion, device, clip_value=1.0):
    model.train()
    epoch_loss = 0
    processed_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Skip empty batches
        if len(batch['label']) == 0:
            continue
            
        history_emb = batch['history_emb'].to(device)
        history_mask = batch['history_mask'].to(device)
        target_emb = batch['target_emb'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(target_emb, history_emb, history_mask)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Check if loss is valid
        if not torch.isnan(loss) and not torch.isinf(loss):
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            processed_batches += 1
        else:
            print("Warning: Found NaN or Inf loss, skipping batch")
    
    # Return average loss over all valid batches
    return epoch_loss / max(1, processed_batches)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    processed_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Skip empty batches
            if len(batch['label']) == 0:
                continue
                
            history_emb = batch['history_emb'].to(device)
            history_mask = batch['history_mask'].to(device)
            target_emb = batch['target_emb'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass (get logits)
            logits, _ = model(target_emb, history_emb, history_mask)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Convert logits to probabilities for metrics
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

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = DINDataset(
        train_user_ids, 
        user_clicked_article_ids, 
        article_embedding_dict, 
        is_training=True
    )
    test_dataset = DINDataset(
        test_user_ids, 
        user_clicked_article_ids, 
        article_embedding_dict, 
        is_training=False
    )
    
    print(f"Generated {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    # Check if we have valid samples
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("ERROR: Not enough valid samples generated. Check data processing.")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128,  # Reduced batch size 
        shuffle=True, 
        num_workers=4,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128,  # Reduced batch size
        shuffle=False, 
        num_workers=4,
        drop_last=False
    )
    
    # Initialize model
    model = DIN(EMBEDDING_DIM, ATTENTION_UNITS).to(device)
    
    # Initialize optimizer with learning rate scheduling and weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
    # Use BCEWithLogitsLoss for numerical stability (combines sigmoid and BCE)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    num_epochs = 10
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 3
    
    print(f"Starting training with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    # Check data distribution
    train_labels = [sample['label'] for sample in train_dataset.samples]
    train_label_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    print(f"Training data distribution: {train_label_counts}")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_din_model.pth')
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping after {epoch+1} epochs without improvement")
            break

    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_din_model.pth'))
    final_loss, final_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Final model performance - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()