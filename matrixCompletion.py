from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch import device, tensor, long, cat, no_grad, save, load
from torch.cuda import is_available
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from os.path import exists

embedding_size = 128
batch_size = 1024
num_epochs = 1
test_steps = 1000
checkpoint = 'mc.pth'
prefix = 'movies'

class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    def __len__(self):
        return len(self.users)
    def __getitem__(self, index):
        return {
            "users": tensor(self.users[index], dtype=long),
            "movies": tensor(self.movies[index], dtype=long),
            "ratings": tensor(self.ratings[index], dtype=long)
        }
    
class RecSysModel(Module):
    def __init__(self, num_users, num_movies, num_rating):
        super().__init__()
        self.user_embedding = Embedding(num_users, embedding_size)
        self.movie_embedding = Embedding(num_movies, embedding_size)
        self.fc = Linear(2 * embedding_size, num_rating)
    def forward(self, users, movies):
        return self.fc(cat([self.user_embedding(users), self.movie_embedding(movies)], dim=1))

def main():
    writer = SummaryWriter('logs')
    gpu = device('cuda' if is_available() else 'cpu')

    df = pd.read_csv(prefix + '/ratings.csv')
    userEncoder = LabelEncoder()
    movieEncoder = LabelEncoder()
    ratingEncoder = LabelEncoder()
    df.userId = userEncoder.fit_transform(df.userId.values)
    df.movieId = movieEncoder.fit_transform(df.movieId.values)
    df.rating = ratingEncoder.fit_transform(df.rating.values)
    df_train, df_valid = train_test_split(df, test_size=0.1, random_state=42, stratify=df.rating.values)
    train_dataset = MovieDataset(users=df_train.userId.values, movies=df_train.movieId.values, ratings=df_train.rating.values)
    valid_dataset = MovieDataset(users=df_valid.userId.values, movies=df_valid.movieId.values, ratings=df_valid.rating.values)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = RecSysModel(len(userEncoder.classes_), len(movieEncoder.classes_), len(ratingEncoder.classes_)).to(gpu)
    if exists(checkpoint):
        model.load_state_dict(load(checkpoint, weights_only=True))
        print('Loaded pretrained weight!')
    optimizer = AdamW(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    loss_function = CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        correct_prediction, train_loss, samples, batch_count = 0, 0, 0, 0
        for i, train_data in enumerate(train_loader):
            users, movies = train_data["users"].to(gpu), train_data["movies"].to(gpu)
            output = model(users, movies)
            rating = train_data["ratings"].to(gpu)
            loss = loss_function(output, rating)
            train_loss += loss.item()
            batch_count += 1
            prediction = output.argmax(dim=1)
            correct_prediction += (prediction == rating).sum().item()
            samples += prediction.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i != 0 and i % test_steps == 0) or i == len(train_loader)-1:
                train_acc = correct_prediction/samples
                train_loss /= batch_count
                model.eval()
                correct_prediction, test_loss = 0, 0
                with no_grad():
                    for valid_data in valid_loader:
                        users = valid_data["users"].to(gpu)
                        movies = valid_data["movies"].to(gpu)
                        output = model(users, movies)
                        rating = valid_data["ratings"].to(gpu)
                        test_loss += loss_function(output, rating).item()
                        prediction = output.argmax(dim=1)
                        correct_prediction += (prediction == rating).sum().item()
                test_acc = correct_prediction/len(valid_dataset)
                test_loss /= len(valid_loader)
                # writer.add_scalars("CF/Accuracy", {"Train": train_acc, "Test": test_acc}, global_step = i + epoch * len(train_loader))
                # writer.add_scalars("CF/Loss", {"Train": train_loss, "Test": test_loss}, global_step = i + epoch * len(train_loader))
                print(f"Epoch [{epoch+1}/{num_epochs}] Steps [{i}/{len(train_loader)-1}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
                correct_prediction, train_loss, samples, batch_count = 0, 0, 0, 0
                model.train()
        scheduler.step()
    writer.close()
    save(model.state_dict(), checkpoint)

if __name__ == '__main__':
    # tensorboard --logdir=logs --port=6007
    # nvidia-smi
    main()